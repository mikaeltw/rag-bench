import os
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str


class RetrieverCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    k: int = Field(4, ge=1, le=100)


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paths: List[str]


class ProviderModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    chat: Dict[str, Any] | None = None
    embeddings: Dict[str, Any] | None = None


class RuntimeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    offline: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: ModelCfg
    retriever: RetrieverCfg
    data: DataCfg
    provider: ProviderModelCfg | None = None
    vector: Dict[str, Any] | None = None
    runtime: RuntimeCfg = RuntimeCfg()


def load_config(path: str) -> BenchConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = os.path.expandvars(fh.read())
    obj: Dict[str, Any] = yaml.safe_load(raw) or {}
    try:
        return BenchConfig.model_validate(obj)
    except ValidationError as e:
        raise SystemExit(f"Invalid config:\n{e}")
