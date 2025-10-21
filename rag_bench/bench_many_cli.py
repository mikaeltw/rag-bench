
import argparse, glob, json, os, yaml
from pathlib import Path
from statistics import mean
from rich.console import Console
from rag_bench.config import load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.eval.metrics import lexical_f1, bow_cosine, context_recall
from rag_bench.providers.base import build_chat_adapter, build_embeddings_adapter
from rag_bench.pipelines import naive_rag, multi_query as mq, rerank as rr, hyde as hy

console = Console()

def choose(cfg_path: str, docs):
    cfg = load_config(cfg_path)
    chat_adapter = build_chat_adapter(cfg.model_dump().get('provider')) if getattr(cfg, 'provider', None) else None
    emb_adapter = build_embeddings_adapter(cfg.model_dump().get('provider')) if getattr(cfg, 'provider', None) else None
    llm_obj = chat_adapter.to_langchain() if chat_adapter else None
    emb_obj = emb_adapter.to_langchain() if emb_adapter else None

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(os.path.expandvars(f.read())) or {}
    if "rerank" in raw:
        rrc = raw["rerank"]
        chain, debug = rr.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k,
                                      rerank_top_k=int(rrc.get("top_k",4)), method=str(rrc.get("method","cosine")),
                                      cross_encoder_model=str(rrc.get("cross_encoder_model","BAAI/bge-reranker-base")),
                                      llm=llm_obj, embeddings=emb_obj)
        return "rerank", chain, debug, cfg
    if "multi_query" in raw:
        n = int(raw["multi_query"].get("n_queries",3))
        chain, debug = mq.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, n_queries=n, llm=llm_obj, embeddings=emb_obj)
        return "multi_query", chain, debug, cfg
    if "hyde" in raw:
        chain, debug = hy.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj)
        return "hyde", chain, debug, cfg
    chain, debug = naive_rag.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj)
    return "naive", chain, debug, cfg

def main():
    ap = argparse.ArgumentParser(description="Run multiple configs and produce a combined HTML report")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--qa", required=True)
    args = ap.parse_args()

    first = sorted(glob.glob(args.configs))[0]
    cfg = load_config(first)
    docs = load_texts_as_documents(cfg.data.paths)

    def iter_jsonl(path):
        with open(path,"r",encoding="utf-8") as f:
            for line in f: yield json.loads(line)

    results = []
    for p in sorted(glob.glob(args.configs)):
        pid, chain, debug, _ = choose(p, docs)
        rows = []
        for ex in iter_jsonl(args.qa):
            q = ex["question"]; ref = ex["reference_answer"]
            ans = chain.invoke(q); dbg = debug()
            retrieved = ""
            if dbg.get("retrieved"): retrieved = "\n".join(r.get("preview","") for r in dbg["retrieved"])
            elif dbg.get("candidates"): retrieved = "\n".join(r.get("preview","") for r in dbg["candidates"][:5])
            m = {"lexical_f1": lexical_f1(ans, ref), "bow_cosine": bow_cosine(ans, ref), "context_recall": context_recall(ref, retrieved) if retrieved else 0.0}
            rows.append(m)
        avg = {k: mean(r[k] for r in rows) if rows else 0.0 for k in ["lexical_f1","bow_cosine","context_recall"]}
        console.print(f"[bold]{Path(p).name} ({pid})[/bold] -> {avg}")
        results.append({"config": Path(p).name, "pipeline": pid, **avg})

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("reports") / f"summary-{ts}.html"; out.parent.mkdir(exist_ok=True, parents=True)
    rows_html = "".join(f"<tr><td>{r['config']}</td><td>{r['pipeline']}</td><td>{r['lexical_f1']:.3f}</td><td>{r['bow_cosine']:.3f}</td><td>{r['context_recall']:.3f}</td></tr>" for r in results)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>rag-bench multi-run</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}</style>
</head><body><h1>rag-bench multi-run summary</h1><table><thead><tr><th>Config</th><th>Pipeline</th><th>Lexical F1</th><th>BoW Cosine</th><th>Context Recall</th></tr></thead><tbody>{rows_html}</tbody></table></body></html>"""
    out.write_text(html, encoding="utf-8")
    console.print(f"[green]Wrote {out}[/green]")

if __name__ == "__main__":
    main()
