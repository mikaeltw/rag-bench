from __future__ import annotations

from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_bench.utils.factories import make_hf_embeddings


def _cosine(u, v):
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un == 0 or vn == 0:
        return 0.0
    return float(np.dot(u, v) / (un * vn))


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 8,
    rerank_top_k: int = 4,
    method: str = "cosine",
    cross_encoder_model: str = "BAAI/bge-reranker-base",
    llm: Optional[object] = None,
    embeddings: Optional[object] = None,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = FAISS.from_documents(splits, embed)

    def build_context(question: str):
        candidates = vect.similarity_search(question, k=k)
        qv = embed.embed_query(question)
        scores = []
        for d in candidates:
            dv = embed.embed_query(d.page_content)
            scores.append((d, _cosine(qv, dv)))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = [d for d, _ in scores[:rerank_top_k]]
        context = "\n\n".join(d.page_content for d in chosen)
        build_context._last_debug = {
            "pipeline": "rerank",
            "method": "cosine",
            "rerank_top_k": rerank_top_k,
            "candidates": [
                {"score": float(sc), "preview": doc.page_content[:160], "source": doc.metadata.get("source", "")}
                for doc, sc in scores[:20]
            ],
        }
        return context

    build_context._last_debug = {"pipeline": "rerank", "method": "unknown", "candidates": []}
    template = (
        "You are a helpful assistant. Use the context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate.from_template(template)
    llm_answer = llm or ChatOpenAI(model=model, temperature=0)

    chain = (
        {"context": RunnableLambda(build_context), "question": RunnablePassthrough()}
        | prompt
        | llm_answer
        | StrOutputParser()
    )

    def debug():
        return getattr(build_context, "_last_debug", {"pipeline": "rerank"})

    return chain, debug
