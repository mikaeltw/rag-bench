
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

def load_texts_as_documents(paths: List[str]) -> List[Document]:
    docs = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": p}))
    return docs
