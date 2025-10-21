
from __future__ import annotations
from typing import List
import os, yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict

class ModelCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str

class RetrieverCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    k: int = Field(4, ge=1, le=100)

class DataCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    paths: List[str]

class ProviderModelCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str
    chat: dict | None = None
    embeddings: dict | None = None

class BenchConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model: ModelCfg
    retriever: RetrieverCfg
    data: DataCfg
    provider: ProviderModelCfg | None = None

def _expand_env(text: str) -> str:
    return os.path.expandvars(text)

def load_config(path: str) -> BenchConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = _expand_env(raw)
    obj = yaml.safe_load(expanded) or {}
    try:
        return BenchConfig.model_validate(obj)
    except ValidationError as e:
        raise SystemExit(f"Invalid config:\n{e}")
