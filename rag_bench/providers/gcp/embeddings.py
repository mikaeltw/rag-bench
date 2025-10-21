from .auth import is_installed
class VertexEmbeddingsAdapter:
    def __init__(self,cfg): self.cfg=cfg
    def to_langchain(self):
        if not is_installed(): raise RuntimeError('Install: rag-bench[gcp]')
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(model_name=self.cfg.get('model','text-embedding-004'), location=self.cfg.get('location','us-central1'), project=self.cfg.get('project_id'))
