from .auth import is_installed


class BedrockEmbeddingsAdapter:
    def __init__(self, root_cfg, cfg):
        self.root = root_cfg
        self.cfg = cfg

    def to_langchain(self):
        if not is_installed():
            raise RuntimeError("Install: rag-bench[aws]")
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            model_id=self.cfg.get("model", "amazon.titan-embed-text-v2:0"),
            region_name=self.root.get("region", "us-east-1"),
        )
