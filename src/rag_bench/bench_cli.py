import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Mapping, Tuple

import yaml
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from rich.console import Console

from rag_bench.config import BenchConfig, load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.eval.metrics import bow_cosine, context_recall, lexical_f1
from rag_bench.eval.report import write_simple_report
from rag_bench.pipelines import hyde as hy
from rag_bench.pipelines import multi_query as mq
from rag_bench.pipelines import naive_rag
from rag_bench.pipelines import rerank as rr
from rag_bench.providers.base import build_chat_adapter, build_embeddings_adapter

console = Console()


def _choose_pipeline(
    cfg_path: str, docs: list[Document]
) -> Tuple[BenchConfig, RunnableSerializable[str, str], Callable[[], Mapping[str, Any]], str]:
    cfg = load_config(cfg_path)
    provider_cfg = cfg.model_dump().get("provider")
    chat_adapter = build_chat_adapter(provider_cfg) if getattr(cfg, "provider", None) else None
    emb_adapter = build_embeddings_adapter(provider_cfg) if getattr(cfg, "provider", None) else None
    llm_obj = chat_adapter.to_langchain() if chat_adapter else None
    emb_obj = emb_adapter.to_langchain() if emb_adapter else None

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(os.path.expandvars(f.read())) or {}
    if "rerank" in raw_cfg:
        rrc = raw_cfg["rerank"]
        chain, debug = rr.build_chain(
            docs,
            model=cfg.model.name,
            k=cfg.retriever.k,
            rerank_top_k=int(rrc.get("top_k", 4)),
            method=str(rrc.get("method", "cosine")),
            cross_encoder_model=str(rrc.get("cross_encoder_model", "BAAI/bge-reranker-base")),
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipe_id = "rerank"
    elif "multi_query" in raw_cfg:
        n = int(raw_cfg["multi_query"].get("n_queries", 3))
        chain, debug = mq.build_chain(
            docs, model=cfg.model.name, k=cfg.retriever.k, n_queries=n, llm=llm_obj, embeddings=emb_obj
        )
        pipe_id = "multi_query"
    elif "hyde" in raw_cfg:
        chain, debug = hy.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj)
        pipe_id = "hyde"
    else:
        chain, debug = naive_rag.build_chain(
            docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj
        )
        pipe_id = "naive"
    return cfg, chain, debug, pipe_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a RAG pipeline on a QA set")
    ap.add_argument("--config", required=True)
    ap.add_argument("--qa", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs = load_texts_as_documents(cfg.data.paths)

    cfg2, chain, debug, pipe_id = _choose_pipeline(args.config, docs)

    rows: list[Dict[str, float]] = []
    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            ref = ex["reference_answer"]
            ans = chain.invoke(q)
            dbg = debug()
            retrieved = ""
            if dbg.get("retrieved"):
                retrieved = "\n".join(r.get("preview", "") for r in dbg["retrieved"])
            elif dbg.get("candidates"):
                retrieved = "\n".join(r.get("preview", "") for r in dbg["candidates"][:5])
            metrics: Dict[str, float] = {
                "lexical_f1": lexical_f1(ans, ref),
                "bow_cosine": bow_cosine(ans, ref),
                "context_recall": context_recall(ref, retrieved) if retrieved else 0.0,
            }
            rows.append(metrics)
            console.print(
                f"[bold cyan]{q}[/bold cyan] -> F1={metrics['lexical_f1']:.3f} "
                f"Cos={metrics['bow_cosine']:.3f} "
                f"Ctx={metrics['context_recall']:.3f}"
            )
    avg: Dict[str, float] = {
        k: mean(r[k] for r in rows) if rows else 0.0 for k in ["lexical_f1", "bow_cosine", "context_recall"]
    }
    console.rule("[bold green]Averages")
    console.print(avg)
    summary: Dict[str, Any] = {"pipeline": pipe_id, "avg_metrics": avg, "num_examples": len(rows)}
    report_path = write_simple_report(
        question=f"Benchmark: {pipe_id} on {Path(args.qa).name}",
        answer=json.dumps(summary, indent=2),
        cfg=cfg2.model_dump(),
        extras={"pipeline": pipe_id},
    )
    console.print(f"[green]Benchmark report written to {report_path}[/green]")


if __name__ == "__main__":
    main()
