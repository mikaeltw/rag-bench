from langchain_core.callbacks.base import BaseCallbackHandler


class UsageTracker(BaseCallbackHandler):
    def __init__(self, cost_per_1k_input=0.0, cost_per_1k_output=0.0):
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0
        self.cpi = cost_per_1k_input
        self.cpo = cost_per_1k_output

    def on_llm_start(self, serialized, prompts, **kw):
        self.in_tok += sum(len(p.split()) for p in prompts)

    def on_llm_end(self, response, **kw):
        self.calls += 1
        try:
            outs = []
            for gens in response.generations:
                for g in gens:
                    outs.append(getattr(g, "text", "") or "")
            self.out_tok += sum(len(t.split()) for t in outs)
        except Exception:
            pass

    def summary(self):
        return {"calls": self.calls, "input_tokens": self.in_tok, "output_tokens": self.out_tok}
