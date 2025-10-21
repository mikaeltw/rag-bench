import random,hashlib,time

def set_seeds(s=42):
    random.seed(s)
    try:
        import numpy as np; np.random.seed(s)
    except Exception: pass

def make_run_id():
    return hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:10]
