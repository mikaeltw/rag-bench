
from __future__ import annotations
from typing import Protocol, Optional, Any, Dict

class ChatAdapter(Protocol):
    def to_langchain(self) -> Any: ...

class EmbeddingsAdapter(Protocol):
    def to_langchain(self) -> Any: ...

def build_chat_adapter(cfg: Dict[str, Any]) -> Optional[ChatAdapter]:
    if not cfg: return None
    name = (cfg.get("name") or "").lower(); chat_cfg = cfg.get("chat", {})
    if name == "gcp":
        from .gcp.chat import VertexChatAdapter; return VertexChatAdapter(chat_cfg)
    if name == "aws":
        from .aws.chat import BedrockChatAdapter; return BedrockChatAdapter(cfg, chat_cfg)
    if name == "azure":
        from .azure.chat import AzureOpenAIChatAdapter; return AzureOpenAIChatAdapter(chat_cfg)
    raise ValueError(f"Unknown provider name: {name}")

def build_embeddings_adapter(cfg: Dict[str, Any]) -> Optional[EmbeddingsAdapter]:
    if not cfg: return None
    name = (cfg.get("name") or "").lower(); emb_cfg = cfg.get("embeddings", {})
    if name == "gcp":
        from .gcp.embeddings import VertexEmbeddingsAdapter; return VertexEmbeddingsAdapter(emb_cfg)
    if name == "aws":
        from .aws.embeddings import BedrockEmbeddingsAdapter; return BedrockEmbeddingsAdapter(cfg, emb_cfg)
    if name == "azure":
        from .azure.embeddings import AzureOpenAIEmbeddingsAdapter; return AzureOpenAIEmbeddingsAdapter(emb_cfg)
    raise ValueError(f"Unknown provider name: {name}")
