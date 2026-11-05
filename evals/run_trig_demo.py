"""Trig demo: blueprint -> generate -> verify -> regenerate failed -> preview + review files."""
import json, sys, time, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.llm import get_provider
from agent.tools.paper import BlueprintArgs, blueprint_defaults, build_and_verify

prov = sys.argv[1] if len(sys.argv) > 1 else "anthropic"
t0 = time.time()
bp = blueprint_defaults(None, BlueprintArgs(class_grade=10, subject="Mathematics", chapter="Introduction to Trigonometry",
                                            total_marks=40, assertion_reason_count=2, case_study_count=1))
out = build_and_verify(get_provider(prov), bp, on_status=lambda k, p: print(k, p, flush=True))
out["blueprint"] = bp
out["latency_s"] = round(time.time() - t0, 1)
out["provider"] = prov
json.dump(out, open(os.path.join(os.path.dirname(__file__), "sample_trigonometry_result.json"), "w", encoding="utf-8"), indent=1, default=str)
s = out["summary"]
print({k: s[k] for k in ("objective_verified", "objective_total", "rubric_checked", "rubric_total", "manual_review", "failed", "marks_sum", "marks_sum_ok", "duplicates", "rounds")}, out["latency_s"], out["usage"])
