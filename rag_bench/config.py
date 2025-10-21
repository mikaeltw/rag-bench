
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import List
import os, yaml
class ModelCfg(BaseModel): model_config=ConfigDict(extra='forbid', strict=True); name:str
class RetrieverCfg(BaseModel): model_config=ConfigDict(extra='forbid', strict=True); k:int=Field(4,ge=1,le=100)
class DataCfg(BaseModel): model_config=ConfigDict(extra='forbid', strict=True); paths:List[str]
class ProviderModelCfg(BaseModel): model_config=ConfigDict(extra='forbid', strict=True); name:str; chat:dict|None=None; embeddings:dict|None=None
class BenchConfig(BaseModel):
    model_config=ConfigDict(extra='forbid', strict=True)
    model:ModelCfg; retriever:RetrieverCfg; data:DataCfg; provider:ProviderModelCfg|None=None; vector:dict|None=None
def load_config(path:str)->BenchConfig:
    raw=os.path.expandvars(open(path,'r',encoding='utf-8').read()); obj=yaml.safe_load(raw) or {}
    try: return BenchConfig.model_validate(obj)
    except ValidationError as e: raise SystemExit(f"Invalid config:\n{e}")
