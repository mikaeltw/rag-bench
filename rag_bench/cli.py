
import argparse, os, yaml
from rich.console import Console
from rag_bench.config import load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.utils.repro import set_seeds, make_run_id
from rag_bench.utils.callbacks.usage import UsageTracker
from rag_bench.utils.cache import cache_get, cache_set
from rag_bench.vector.base import build_vector_backend
from rag_bench.providers.base import build_chat_adapter, build_embeddings_adapter
from rag_bench.pipelines import naive_rag

console = Console()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--question", required=True)
    args = ap.parse_args(); set_seeds(42); run_id=make_run_id()
    cfg = load_config(args.config); docs = load_texts_as_documents(cfg.data.paths)
    chat = build_chat_adapter(cfg.model_dump().get('provider')) if getattr(cfg, 'provider', None) else None
    emb = build_embeddings_adapter(cfg.model_dump().get('provider')) if getattr(cfg, 'provider', None) else None
    llm_obj = chat.to_langchain() if chat else None; emb_obj = emb.to_langchain() if emb else None
    vec = build_vector_backend(cfg.model_dump().get('vector')); retr=None
    if vec and emb_obj is not None:

        try: retr = vec.make_retriever(docs=None, embeddings=emb_obj, k=cfg.retriever.k)
        except Exception: retr=None
    chain,_ = naive_rag.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb_obj, retriever=retr)
    prompt=args.question; cached=cache_get(cfg.model.name, prompt)
    if cached is None:

        ans = chain.invoke(prompt, config={'callbacks':[UsageTracker()]}); cache_set(cfg.model.name, prompt, ans)
    else:
        ans=cached
    console.print(ans)
if __name__ == "__main__": main()
