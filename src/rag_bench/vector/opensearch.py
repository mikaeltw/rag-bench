def _require():
    try:
        from langchain_community.vectorstores import OpenSearchVectorSearch

        return OpenSearchVectorSearch
    except Exception as e:
        raise RuntimeError("OpenSearch requires opensearch-py (install rag-bench[aws])") from e


class OpenSearchBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def make_retriever(self, *, docs, embeddings, k: int):
        V = _require()
        hosts = self.cfg.get("hosts")
        idx = self.cfg.get("index")
        if not hosts or not idx:
            raise ValueError("hosts/index required")
        vs = V(
            index_name=idx,
            embedding_function=embeddings,
            hosts=hosts,
            use_ssl=self.cfg.get("use_ssl", True),
            verify_certs=self.cfg.get("verify_certs", True),
            http_auth=self.cfg.get("http_auth"),
        )
        return vs.as_retriever(search_kwargs={"k": k})
