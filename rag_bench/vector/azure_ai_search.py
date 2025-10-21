def _require():
    try:
        from langchain_community.vectorstores.azuresearch import AzureSearch
        return AzureSearch
    except Exception as e:
        raise RuntimeError('Azure AI Search requires azure-search-documents (install rag-bench[azure])') from e
class AzureAISearchBackend:
    def __init__(self,cfg): self.cfg=cfg
    def make_retriever(self,*,docs,embeddings,k:int):
        AzureSearch=_require(); ep=self.cfg.get('endpoint'); idx=self.cfg.get('index'); key=self.cfg.get('api_key')
        if not ep or not idx: raise ValueError('endpoint/index required')
        vs=AzureSearch(azure_search_endpoint=ep, azure_search_key=key, index_name=idx, embedding_function=embeddings)
        return vs.as_retriever(search_kwargs={'k':k})
