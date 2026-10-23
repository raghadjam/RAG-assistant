"""Fakes used by the Week 3 tests, so nothing needs the network or an API key."""

import zlib

import numpy as np

from src.llm_client import LLMResponse


class HashEmbedder:
    """Words that two texts share give similar vectors, with no model download."""
    dim = 256

    def embed(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, zlib.crc32(word.encode()) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeResponse:
    """Looks enough like requests.Response for the LLM client."""

    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def ok(text, model="served-model", finish="stop", prompt_tokens=12, completion_tokens=7):
    return FakeResponse(200, {
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


def err(status, message="something went wrong", headers=None):
    return FakeResponse(status, {"error": {"code": status, "message": message}}, headers=headers)


class FakeSession:
    """Plays back a script of responses (or exceptions to raise), one per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    """Stands in for LLMClient in pipeline tests. Replies are strings, LLMResponses or exceptions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, max_tokens=None):
        self.calls.append(messages)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=item, model="fake-model", prompt_tokens=10, completion_tokens=5)
