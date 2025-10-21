from .auth import is_installed
class VertexChatAdapter:
    def __init__(self,cfg): self.cfg=cfg
    def to_langchain(self):
        if not is_installed(): raise RuntimeError('Install: rag-bench[gcp]')
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(model=self.cfg.get('model','gemini-1.5-pro'), location=self.cfg.get('location','us-central1'), project=self.cfg.get('project_id'), temperature=0)
