from collections import Counter
_tok=lambda s:[t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

def lexical_f1(p,r):
    P=_tok(p); R=_tok(r)
    if not P or not R:return 0.0
    Pc, Rc = Counter(P), Counter(R)
    ov=sum(min(Pc[t],Rc[t]) for t in set(Pc)|set(Rc))
    pr=ov/max(1,sum(Pc.values())); rc=ov/max(1,sum(Rc.values()))
    return 0.0 if pr+rc==0 else 2*pr*rc/(pr+rc)

def bow_cosine(p,r):
    P=Counter(_tok(p)); R=Counter(_tok(r))
    if not P or not R: return 0.0
    keys=set(P)|set(R); dp=sum(P[k]*R[k] for k in keys)
    import math
    return 0.0 if not P or not R else dp/(math.sqrt(sum(v*v for v in P.values()))*math.sqrt(sum(v*v for v in R.values())))
