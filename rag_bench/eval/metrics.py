
from __future__ import annotations
from collections import Counter
from typing import List

def _tokens(s: str) -> List[str]:
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

def lexical_f1(pred: str, ref: str) -> float:
    p = _tokens(pred); r = _tokens(ref)
    if not p or not r: return 0.0
    pc, rc = Counter(p), Counter(r)
    overlap = sum(min(pc[t], rc[t]) for t in set(pc) | set(rc))
    prec = overlap / max(1, sum(pc.values())); rec = overlap / max(1, sum(rc.values()))
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def bow_cosine(pred: str, ref: str) -> float:
    p = Counter(_tokens(pred)); r = Counter(_tokens(ref))
    if not p or not r: return 0.0
    keys = set(p)|set(r)
    dp = sum(p[k]*r[k] for k in keys)
    pn = sum(v*v for v in p.values())**0.5; rn = sum(v*v for v in r.values())**0.5
    return 0.0 if pn==0 or rn==0 else dp/(pn*rn)

def context_recall(reference: str, retrieved_text: str) -> float:
    ref_tokens = _tokens(reference)
    from collections import Counter
    counts = Counter(t for t in ref_tokens if len(t)>2)
    if not counts: return 0.0
    top = [t for t,_ in counts.most_common(10)]
    hits = sum(1 for t in top if t in set(_tokens(retrieved_text)))
    return hits/len(top)
