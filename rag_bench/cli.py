
import argparse, os, yaml
from rich.console import Console
from rag_bench.config import load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.eval.report import write_simple_report
from rag_bench.utils.repro import set_seeds, make_run_id
from rag_bench.utils.callbacks.usage import UsageTracker
from rag_bench.utils.cache import cache_get, cache_set
from rag_bench.providers.base import build_chat_adapter, build_embeddings_adapter

from rag_bench.pipelines import naive_rag
from rag_bench.pipelines import multi_query as mq
from rag_bench.pipelines import rerank as rr
from rag_bench.pipelines import hyde as hy

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Run rag-bench pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cost-in", type=float, default=0.0)
    parser.add_argument("--cost-out", type=float, default=0.0)
    args = parser.parse_args()

    set_seeds(args.seed)
    run_id = args.run_id or make_run_id()

    cfg = load_config(args.config)
    docs = load_texts_as_documents(cfg.data.paths)

    # provider adapters (optional)
    chat_adapter = build_chat_adapter(cfg.model_dump().get('provider')) if getattr(cfg, 'provider', None) else None
    emb_adapter = build_embeddings_adapter(cfg.model_dump().get('provider')) if getattr(cfg, 'provider', None) else None
    llm_obj = chat_adapter.to_langchain() if chat_adapter else None
    emb_obj = emb_adapter.to_langchain() if emb_adapter else None

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(os.path.expandvars(f.read())) or {}

    if "rerank" in raw_cfg:
        rrc = raw_cfg["rerank"]
        chain, debug = rr.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k,
                                      rerank_top_k=int(rrc.get("top_k",4)), method=str(rrc.get("method","cosine")),
                                      cross_encoder_model=str(rrc.get("cross_encoder_model","BAAI/bge-reranker-base")),
                                      llm=llm_obj, embeddings=emb_obj)
    elif "multi_query" in raw_cfg:
        n_queries = int(raw_cfg["multi_query"].get("n_queries",3))
        chain, debug = mq.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, n_queries=n_queries, llm=llm_obj, embeddings=emb_obj)
    elif "hyde" in raw_cfg:
        chain, debug = hy.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj)
    else:
        chain, debug = naive_rag.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj)

    console.rule(f"[bold]Running pipeline (run_id={run_id})")
    tracker = UsageTracker(cost_per_1k_input=args.cost_in, cost_per_1k_output=args.cost_out)
    prompt = args.question
    cached = cache_get(cfg.model.name, prompt)
    if cached is not None:
        answer = cached; console.print("[yellow]Loaded answer from cache[/yellow]")
    else:
        answer = chain.invoke(prompt, config={"callbacks":[tracker]})
        cache_set(cfg.model.name, prompt, answer)

    console.print(f"[bold]Answer:[/bold] {answer}")
    extras = debug(); extras["usage"] = tracker.summary(); extras["run_id"] = run_id
    report_path = write_simple_report(question=args.question, answer=answer, cfg=cfg.model_dump(), extras=extras)
    console.print(f"[green]Report written to {report_path}[/green]")

if __name__ == "__main__":
    main()
