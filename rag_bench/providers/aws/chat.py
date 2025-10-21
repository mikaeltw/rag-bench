from .auth import is_installed
class BedrockChatAdapter:
    def __init__(self,root_cfg,cfg): self.root=root_cfg; self.cfg=cfg
    def to_langchain(self):
        if not is_installed(): raise RuntimeError('Install: rag-bench[aws]')
        from langchain_aws import ChatBedrock
        return ChatBedrock(model_id=self.cfg.get('model','anthropic.claude-3-5-sonnet-20240620-v1:0'), region_name=self.root.get('region','us-east-1'), temperature=0)
