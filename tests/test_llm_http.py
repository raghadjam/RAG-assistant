"""LLMClient against a real HTTP server running on localhost (no internet needed).

The server plays a script of responses, so real timeouts, real status codes and a
real refused connection are exercised, not just fakes.
"""

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.llm_client import LLMAuthError, LLMClient, LLMUnavailableError  # noqa: E402

MESSAGES = [{"role": "user", "content": "hi"}]


def completion(text="Hello [1]."):
    return {"model": "served/model", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4}}


class ScriptedServer:
    def __init__(self, script):
        # each step: (status, body, headers, delay_seconds); body is a dict or raw string
        self.script = list(script)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "json": json.loads(raw)})
                status, body, headers, delay = outer.script.pop(0)
                time.sleep(delay)
                payload = body if isinstance(body, str) else json.dumps(body)
                try:
                    self.send_response(status)
                    for key, value in (headers or {}).items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload.encode())))
                    self.end_headers()
                    self.wfile.write(payload.encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(script):
        server = ScriptedServer(script)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def make_client(base_url, **overrides):
    options = {"api_key": "secret-key", "base_url": base_url, "models": ("m1",), "max_retries": 2,
               "timeout_s": 5.0, **overrides}
    session = requests.Session()
    session.trust_env = False        # ignore any proxy settings for localhost
    return LLMClient(Settings(**options), session=session, sleep=lambda s: None, jitter=lambda a, b: 0)


def test_real_round_trip(serve):
    server = serve([(200, completion(), None, 0)])
    reply = make_client(server.url).chat(MESSAGES, max_tokens=50)
    assert reply.text == "Hello [1]."
    assert reply.model == "served/model"
    seen = server.requests[0]
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["json"]["messages"] == MESSAGES and seen["json"]["max_tokens"] == 50


def test_rate_limit_with_retry_after_then_success(serve):
    server = serve([(429, {"error": {"code": 429, "message": "slow down"}}, {"Retry-After": "0"}, 0),
                    (200, completion(), None, 0)])
    reply = make_client(server.url).chat(MESSAGES)
    assert reply.attempts == 2
    assert len(server.requests) == 2


def test_real_read_timeout_is_retried(serve):
    server = serve([(200, completion("too late"), None, 1.0), (200, completion("on time"), None, 0)])
    reply = make_client(server.url, timeout_s=0.3).chat(MESSAGES)
    assert reply.text == "on time"
    assert reply.failures == ["m1: timeout"]


def test_refused_connection_ends_as_unavailable():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]          # nothing listens here once the socket closes
    client = make_client(f"http://127.0.0.1:{port}/v1", max_retries=1)
    with pytest.raises(LLMUnavailableError) as info:
        client.chat(MESSAGES)
    assert "connection error" in info.value.reasons[0]


def test_unauthorized_stops_after_one_request(serve):
    server = serve([(401, {"error": {"code": 401, "message": "No auth credentials found"}}, None, 0)])
    with pytest.raises(LLMAuthError):
        make_client(server.url).chat(MESSAGES)
    assert len(server.requests) == 1


def test_html_error_page_is_retried(serve):
    server = serve([(502, "<html>Bad Gateway</html>", {"Content-Type": "text/html"}, 0),
                    (200, completion(), None, 0)])
    assert make_client(server.url).chat(MESSAGES).text == "Hello [1]."


def test_free_model_returns_error_inside_a_200_response(serve):
    server = serve([(200, {"error": {"code": 503, "message": "provider overloaded"}}, None, 0),
                    (200, completion(), None, 0)])
    assert make_client(server.url).chat(MESSAGES).attempts == 2
