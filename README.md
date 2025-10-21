
# rag-bench

[![CI](https://img.shields.io/github/actions/workflow/status/mikaeltw/rag-bench/ci.yml?branch=main)](../../actions)
[![PyPI](https://img.shields.io/pypi/v/rag-bench.svg)](https://pypi.org/project/rag-bench/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reproducible **RAG** baselines + evaluations, powered by **LangChain**. Configure a pipeline in YAML, run one command, get an HTML report.

## Install
```bash
pip install rag-bench
# or
uv pip install rag-bench
```

## Quickstart
```bash
python run.py --config configs/wiki.yaml --question "What is LangChain?"
python run.py --config configs/multi_query.yaml --question "What is LangChain?"
python run.py --config configs/rerank.yaml --question "What is LangChain?"
python run.py --config configs/hyde.yaml --question "What is LangChain?"
```

## Cloud adapters (optional)
```bash
pip install "rag-bench[gcp]"    # Vertex AI
pip install "rag-bench[aws]"    # Bedrock
pip install "rag-bench[azure]"  # Azure OpenAI
# Then:
rag-bench --config configs/providers/azure.yaml --question "What is LangChain?"
```

## Eval harness
```bash
rag-bench-bench --config configs/multi_query.yaml --qa examples/qa/toy.jsonl
rag-bench-many --configs "configs/*.yaml" --qa examples/qa/toy.jsonl
```

## Contributing
- `make fmt && make lint && make test`
- Open PRs against `main`

See `RELEASE.md` for publishing steps.

### Vector backends (optional)

Use a managed vector store instead of local FAISS by adding a `vector:` block:

```yaml
vector:
  name: azure_ai_search
  endpoint: https://<your>.search.windows.net
  index: my-index
  # api_key: ${AZURE_SEARCH_API_KEY}
```

Other options:
- `opensearch`: `hosts`, `index` (plus IAM/http_auth).
- `matching_engine` (GCP): `project_id`, `location`, `index_id`, `endpoint_id`.

Install extras:
```bash
pip install "rag-bench[azure]"  # for Azure AI Search
pip install "rag-bench[aws]"    # for OpenSearch
pip install "rag-bench[gcp]"    # for Matching Engine
```
