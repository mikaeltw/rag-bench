
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
