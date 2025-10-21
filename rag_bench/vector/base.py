
from typing import Protocol, Any, Dict, Optional, List
from langchain.docstore.document import Document
class VectorBackend(Protocol):
    def make_retriever(self, *, docs: Optional[List[Document]], embeddings: Any, k: int): ...
def build_vector_backend(cfg: Dict[str, Any] | None) -> Optional[VectorBackend]:
    if not cfg: return None
    name=(cfg.get("name") or "").lower()
    if name=="azure_ai_search":
        from .azure_ai_search import AzureAISearchBackend; return AzureAISearchBackend(cfg)
    if name=="opensearch":
        from .opensearch import OpenSearchBackend; return OpenSearchBackend(cfg)
    if name=="matching_engine":
        from .matching_engine import MatchingEngineBackend; return MatchingEngineBackend(cfg)
    raise ValueError(f"Unknown vector backend: {name}")
