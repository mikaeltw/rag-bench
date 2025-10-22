from .auth import is_installed


class AzureOpenAIEmbeddingsAdapter:
    def __init__(self, cfg):
        self.cfg = cfg

    def to_langchain(self):
        if not is_installed():
            raise RuntimeError("Install: rag-bench[azure]")
        from langchain_openai import AzureOpenAIEmbeddings

        dep = self.cfg.get("deployment", "text-embedding-3-large")
        endpoint = self.cfg.get("endpoint")
        ver = self.cfg.get("api_version", "2024-06-01")
        if not endpoint:
            raise ValueError("Azure OpenAI requires endpoint")
        return AzureOpenAIEmbeddings(azure_deployment=dep, azure_endpoint=endpoint, api_version=ver)
