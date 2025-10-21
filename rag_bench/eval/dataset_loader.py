from pathlib import Path
from typing import List
from langchain.docstore.document import Document

def load_texts_as_documents(paths: List[str]):
    docs=[]
    for p in paths:
        docs.append(Document(page_content=Path(p).read_text(encoding='utf-8'), metadata={'source':p}))
    return docs
