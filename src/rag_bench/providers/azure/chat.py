from .auth import is_installed


class AzureOpenAIChatAdapter:
    def __init__(self, cfg):
        self.cfg = cfg

    def to_langchain(self):
        if not is_installed():
            raise RuntimeError("Install: rag-bench[azure]")
        from langchain_openai import AzureChatOpenAI

        dep = self.cfg.get("deployment", "gpt-4o-mini")
        endpoint = self.cfg.get("endpoint")
        ver = self.cfg.get("api_version", "2024-06-01")
        if not endpoint:
            raise ValueError("Azure OpenAI requires endpoint")
        return AzureChatOpenAI(azure_deployment=dep, azure_endpoint=endpoint, api_version=ver, temperature=0)
