
from __future__ import annotations
from typing import Any, Dict
from .auth import is_installed

class AzureOpenAIChatAdapter:
    def __init__(self, chat_cfg: Dict[str, Any]): self.cfg = chat_cfg
    def to_langchain(self):
        if not is_installed(): raise RuntimeError('Azure adapter requires extra: pip install "rag-bench[azure]"')
        from langchain_openai import AzureChatOpenAI
        dep = self.cfg.get("deployment","gpt-4o-mini"); endpoint = self.cfg.get("endpoint"); ver=self.cfg.get("api_version","2024-06-01")
        if not endpoint: raise ValueError("Azure OpenAI requires 'endpoint'")
        return AzureChatOpenAI(azure_deployment=dep, azure_endpoint=endpoint, api_version=ver, temperature=0)
