def _require():
    try:
        from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

        return VectorSearchVectorStore
    except Exception as e:
        raise RuntimeError("Matching Engine requires langchain-google-vertexai (install rag-bench[gcp])") from e


class MatchingEngineBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def make_retriever(self, *, docs, embeddings, k: int):
        V = _require()
        proj = self.cfg.get("project_id")
        loc = self.cfg.get("location", "us-central1")
        idx = self.cfg.get("index_id")
        ep = self.cfg.get("endpoint_id")
        if not (proj and idx and ep):
            raise ValueError("project_id/index_id/endpoint_id required")
        vs = V.from_components(project_id=proj, region=loc, index_id=idx, endpoint_id=ep, embedding=embeddings)
        return vs.as_retriever(search_kwargs={"k": k})
