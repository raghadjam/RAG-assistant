106 queries against an index of 24,368 chunks.

K is the number of chunks returned. A chunk is relevant when its paper is in the query's relevant set.

- **Precision@K**: relevant chunks among the top K, divided by K.
- **Recall@K**: distinct relevant papers found in the top K, divided by min(number of relevant papers, K). For a query with one relevant paper this is the hit rate (was the paper found in the top K).
- **MRR**: mean of 1 / (rank of the first relevant chunk); 0 when none was retrieved.

### Results by query type

| Query type | Queries | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | Latency mean / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| title (known paper) | 40 | 1.00 | 0.69 | 0.49 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 24.2 / 27.1 |
| sentence (known paper) | 40 | 0.78 | 0.52 | 0.36 | 0.20 | 0.78 | 0.82 | 0.85 | 0.85 | 0.81 | 12.5 / 14.5 |
| concept (many papers) | 26 | 0.96 | 0.85 | 0.81 | 0.79 | 0.96 | 0.80 | 0.73 | 0.66 | 0.97 | 12.7 / 16.8 |
| **all queries** | 106 | 0.91 | 0.66 | 0.52 | 0.37 | 0.91 | 0.88 | 0.88 | 0.86 | 0.92 | 17.0 / 21.9 |

### Searching with the paper's own topic as a filter

Each known-paper query is searched twice: without a filter, and restricted to the topic of the paper it should find.

| Query type and setting | Queries | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | Latency mean / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| title (known paper): no filter | 40 | 1.00 | 0.69 | 0.49 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 24.2 / 27.1 |
| title (known paper): topic filter | 40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 24.0 / 30.7 |
| sentence (known paper): no filter | 40 | 0.78 | 0.52 | 0.36 | 0.20 | 0.78 | 0.82 | 0.85 | 0.85 | 0.81 | 12.5 / 14.5 |
| sentence (known paper): topic filter | 40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 14.8 / 16.8 |

### Queries whose paper was not found in the top 10

6 of 106 queries.

| Type | Query | Relevant papers |
|---|---|---|
| sentence | Its speedup, however, can fade as context grows and draft acceptance declines. | 1 |
| sentence | Existing synthesizers automate this task but require users to state their intent as inp... | 1 |
| sentence | However, how it reshapes their internal mechanisms remains poorly understood. | 1 |
| sentence | Each output is classified as Literal, Canonical, or Other, separating faithful transcri... | 1 |
| sentence | Trustworthy scientific agents need deterministic validation of execution, not model-sid... | 1 |
| sentence | Existing approaches align observed representations, model missingness, or reconstruct a... | 1 |
