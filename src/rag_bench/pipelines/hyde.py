from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_bench.utils.factories import make_hf_embeddings

HYP_PROMPT = """You will draft a hypothetical answer to help retrieve relevant passages.
Question: {question}
Draft a concise, factual paragraph:"""


def _fallback_hypothesis(question: str) -> str:
    return f"This is a draft answer about: {question}. It outlines likely definitions, key concepts, and use cases."


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 4,
    llm: Optional[object] = None,
    embeddings: Optional[object] = None,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = FAISS.from_documents(splits, embed)

    openai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    if openai_ok and llm is None:
        llm_h = ChatOpenAI(model=model, temperature=0)
        hyp_tmpl = PromptTemplate.from_template(HYP_PROMPT)

        def gen_hyp(q: str) -> str:
            return (hyp_tmpl | llm_h | StrOutputParser()).invoke({"question": q}).strip()

    else:

        def gen_hyp(q: str) -> str:
            return _fallback_hypothesis(q)

    llm_answer = llm or ChatOpenAI(model=model, temperature=0)

    def build_context(question: str) -> str:
        hyp = gen_hyp(question)
        docs_h = vect.similarity_search(hyp, k=k)
        context = "\n\n".join(d.page_content for d in docs_h)
        build_context._last_debug = {
            "pipeline": "hyde",
            "hypothesis": hyp,
            "retrieved": [{"source": d.metadata.get("source", ""), "preview": d.page_content[:160]} for d in docs_h],
        }
        return context

    build_context._last_debug = {"pipeline": "hyde", "hypothesis": "", "retrieved": []}
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
        return getattr(build_context, "_last_debug", {"pipeline": "hyde"})

    return chain, debug
