
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(".ragbench_cache"); CACHE_DIR.mkdir(exist_ok=True, parents=True)
def _key(model: str, prompt: str) -> str: return hashlib.sha256((model+"||"+prompt).encode()).hexdigest()
def cache_get(model: str, prompt: str):
    p = CACHE_DIR / (_key(model,prompt)+".json")
    if p.exists(): 
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return None
    return None
def cache_set(model: str, prompt: str, output: Any):
    p = CACHE_DIR / (_key(model,prompt)+".json")
    p.write_text(json.dumps(output), encoding="utf-8")
