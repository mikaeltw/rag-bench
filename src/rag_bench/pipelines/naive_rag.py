from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_bench.utils.factories import make_hf_embeddings


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 4,
    llm: Optional[object] = None,
    embeddings: Optional[object] = None,
    retriever: Optional[object] = None,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if retriever is None:
        vect = FAISS.from_documents(splits, embed)
        retr = vect.as_retriever(search_kwargs={"k": k})
    else:
        retr = retriever
    prompt = PromptTemplate.from_template(
        "Use the context to answer.\n" "Context:\n{context}\n\n" "Question: {question}\n" "Answer (end with ###END):"
    )
    llm = llm or ChatOpenAI(model=model, temperature=0)
    llm = llm.bind(stop=["###END"])

    def ctx_join(d):
        return "\n\n".join(x.page_content for x in d)

    chain = {"context": retr | ctx_join, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain, (lambda: {"pipeline": "naive_rag"})
