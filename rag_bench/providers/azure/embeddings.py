
from __future__ import annotations
from typing import Any, Dict
from .auth import is_installed

class AzureOpenAIEmbeddingsAdapter:
    def __init__(self, emb_cfg: Dict[str, Any]): self.cfg = emb_cfg
    def to_langchain(self):
        if not is_installed(): raise RuntimeError('Azure adapter requires extra: pip install "rag-bench[azure]"')
        from langchain_openai import AzureOpenAIEmbeddings
        dep = self.cfg.get("deployment","text-embedding-3-large"); endpoint=self.cfg.get("endpoint"); ver=self.cfg.get("api_version","2024-06-01")
        if not endpoint: raise ValueError("Azure OpenAI requires 'endpoint'")
        return AzureOpenAIEmbeddings(azure_deployment=dep, azure_endpoint=endpoint, api_version=ver)
