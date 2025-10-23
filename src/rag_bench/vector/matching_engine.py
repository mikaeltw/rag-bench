from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Type, cast

if TYPE_CHECKING:
    from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


def _require() -> Type["VectorSearchVectorStore"]:
    try:
        from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

        return cast(Type["VectorSearchVectorStore"], VectorSearchVectorStore)
    except Exception as e:
        raise RuntimeError("Matching Engine requires langchain-google-vertexai (install rag-bench[gcp])") from e


class MatchingEngineBackend:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def make_retriever(
        self,
        *,
        docs: Optional[List[Document]],
        embeddings: Embeddings,
        k: int,
    ) -> VectorStoreRetriever:
        VectorSearchVectorStore = _require()
        proj = self.cfg.get("project_id")
        loc = self.cfg.get("location", "us-central1")
        idx = self.cfg.get("index_id")
        ep = self.cfg.get("endpoint_id")
        if not (proj and idx and ep):
            raise ValueError("project_id/index_id/endpoint_id required")
        vs = VectorSearchVectorStore.from_components(
            project_id=str(proj),
            region=str(loc),
            index_id=str(idx),
            endpoint_id=str(ep),
            embedding=embeddings,
        )
        return cast(VectorStoreRetriever, vs.as_retriever(search_kwargs={"k": k}))
