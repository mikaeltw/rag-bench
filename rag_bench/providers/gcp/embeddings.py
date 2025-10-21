
from __future__ import annotations
from typing import Any, Dict
from .auth import is_installed

class VertexEmbeddingsAdapter:
    def __init__(self, emb_cfg: Dict[str, Any]):
        self.cfg = emb_cfg
    def to_langchain(self):
        if not is_installed(): raise RuntimeError('GCP adapter requires extra: pip install "rag-bench[gcp]"')
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(model_name=self.cfg.get("model","text-embedding-004"),
                                  location=self.cfg.get("location","us-central1"),
                                  project=self.cfg.get("project_id"))
