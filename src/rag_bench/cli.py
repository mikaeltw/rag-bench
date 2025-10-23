import argparse
import os

from rich.console import Console

from rag_bench.config import load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.pipelines import naive_rag
from rag_bench.providers.base import build_embeddings_adapter
from rag_bench.utils.cache import cache_get, cache_set
from rag_bench.utils.callbacks.usage import UsageTracker
from rag_bench.utils.repro import set_seeds
from rag_bench.vector.base import build_vector_backend

console = Console()


def _pick_llm(cfg):
    """Return a LangChain LLM object based on offline flag."""
    if getattr(cfg.runtime, "offline", False):
        # Local, CPU-friendly
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        # small, CPU-runner model
        model_id = "distilgpt2"
        tok = AutoTokenizer.from_pretrained(model_id)
        # make sure we have a pad token for batch/text generation
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        gen = pipeline(
            task="text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_id),
            tokenizer=tok,
            device=-1,  # CPU
            return_full_text=False,  # <-- don't echo the prompt
        )

        # deterministic, short answers, and safe for gpt2-family
        gen.model.generation_config.update(
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        llm = HuggingFacePipeline(pipeline=gen)
        # Important: stop when the model tries to start a new Q/A block
        llm = llm.bind(stop=["\nQuestion:"])
        return llm
    else:
        # Cloud (OpenAI via langchain-openai)
        from rag_bench.providers.base import build_chat_adapter

        prov = getattr(cfg, "provider", None)
        if prov:
            chat = build_chat_adapter(cfg.model_dump().get("provider"))
            return chat.to_langchain()
        else:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=cfg.model.name, temperature=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    set_seeds(42)

    cfg = load_config(args.config)

    # Apply device preference early so torch/embeddings respect it
    dev = getattr(cfg.runtime, "device", "auto")
    if dev == "cpu":
        os.environ.setdefault("RAG_BENCH_DEVICE", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    elif dev in ("cuda",):
        os.environ.setdefault("RAG_BENCH_DEVICE", "cuda")

    docs = load_texts_as_documents(cfg.data.paths)

    # Embeddings: if you’re using the factory, it respects CPU/GPU globally.
    emb = None
    prov = getattr(cfg, "provider", None)
    if prov:
        adapter = build_embeddings_adapter(cfg.model_dump().get("provider"))
        emb = adapter.to_langchain()

    # Vector retriever (optional; safe fallback)
    vec = build_vector_backend(cfg.model_dump().get("vector"))
    retr = None
    if vec and emb is not None:
        try:
            retr = vec.make_retriever(docs=None, embeddings=emb, k=cfg.retriever.k)
        except Exception:
            retr = None

    # Select LLM by offline flag
    llm_obj = _pick_llm(cfg)

    chain, _meta = naive_rag.build_chain(
        docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb, retriever=retr
    )

    prompt = args.question
    cached = cache_get(cfg.model.name, prompt)
    if cached is None:
        ans = chain.invoke(prompt, config={"callbacks": [UsageTracker()]})
        cache_set(cfg.model.name, prompt, ans)
    else:
        ans = cached

    console.print(ans)


if __name__ == "__main__":
    main()
