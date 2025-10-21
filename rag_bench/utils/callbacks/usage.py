
from __future__ import annotations
from typing import Any, Dict
from langchain_core.callbacks.base import BaseCallbackHandler

class UsageTracker(BaseCallbackHandler):
    def __init__(self, cost_per_1k_input: float = 0.0, cost_per_1k_output: float = 0.0):
        self.calls=0; self.input_tokens=0; self.output_tokens=0
        self.cost_in=0.0; self.cost_out=0.0
        self.cpi=cost_per_1k_input; self.cpo=cost_per_1k_output
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        n = sum(len(p.split()) for p in prompts); self.input_tokens += n; self.cost_in += (n/1000.0)*self.cpi
    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.calls += 1
        try:
            texts=[]; 
            for gens in response.generations:
                for g in gens: texts.append(getattr(g,"text","") or "")
            n = sum(len(t.split()) for t in texts); self.output_tokens += n; self.cost_out += (n/1000.0)*self.cpo
        except Exception: pass
    def summary(self)->Dict[str,Any]:
        return {"calls":self.calls,"input_tokens":self.input_tokens,"output_tokens":self.output_tokens,
                "approx_cost_in":round(self.cost_in,6),"approx_cost_out":round(self.cost_out,6),
                "approx_cost_total":round(self.cost_in+self.cost_out,6)}
