from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_bench.utils.factories import make_hf_embeddings

GEN_PROMPT = """You are an expert at generating diverse search queries.
Produce {n} different queries that could retrieve context to answer the user's question.
Return one query per line, no numbering.

Question: {question}
"""


def _fallback_queries(question: str, n: int):
    variants = [
        question,
        f"Background for: {question}",
        f"Key facts related to: {question}",
        f"Overview: {question}",
        f"ELI5: {question}",
    ]
    return variants[: max(1, n)]


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 4,
    n_queries: int = 3,
    llm: Optional[object] = None,
    embeddings: Optional[object] = None,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = FAISS.from_documents(splits, embed)

    llm_answer = llm or ChatOpenAI(model=model, temperature=0)
    openai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    if openai_ok and llm is None:
        llm_gen = ChatOpenAI(model=model, temperature=0)
        gen_tmpl = PromptTemplate.from_template(GEN_PROMPT)

        def gen_queries(q: str):
            text = (gen_tmpl | llm_gen | StrOutputParser()).invoke({"n": n_queries, "question": q})
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            uniq = []
            for s in [q] + lines:
                if s not in uniq:
                    uniq.append(s)
                if len(uniq) >= max(1, n_queries):
                    break
            return uniq

    else:

        def gen_queries(q: str):
            return _fallback_queries(q, n_queries)

    def build_context(question: str) -> str:
        queries = gen_queries(question)
        seen = set()
        aggregated = []
        for qr in queries:
            docs_q = vect.similarity_search(qr, k=k)
            for d in docs_q:
                key = d.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    aggregated.append(d)
        context = "\n\n".join(d.page_content for d in aggregated[: max(k, len(aggregated))])
        build_context._last_debug = {
            "pipeline": "multi_query",
            "queries": queries,
            "retrieved": [
                {"source": d.metadata.get("source", ""), "preview": d.page_content[:160]} for d in aggregated
            ],
        }
        return context

    build_context._last_debug = {"pipeline": "multi_query", "queries": [], "retrieved": []}
    template = (
        "You are a helpful assistant. Use the context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": RunnableLambda(build_context), "question": RunnablePassthrough()}
        | prompt
        | llm_answer
        | StrOutputParser()
    )

    def debug():
        return getattr(build_context, "_last_debug", {"pipeline": "multi_query"})

    return chain, debug
