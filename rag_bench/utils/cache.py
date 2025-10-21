import hashlib,json
from pathlib import Path
D=Path('.ragbench_cache');D.mkdir(exist_ok=True,parents=True)
K=lambda m,p: hashlib.sha256((m+'||'+p).encode()).hexdigest()

def cache_get(m,p):
    f=D/(K(m,p)+'.json')
    if f.exists():
        try: return json.loads(f.read_text('utf-8'))
        except Exception: return None
    return None

def cache_set(m,p,o):
    f=D/(K(m,p)+'.json'); f.write_text(json.dumps(o), 'utf-8')
