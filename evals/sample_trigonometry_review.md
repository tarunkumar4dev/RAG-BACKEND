# Trigonometry demo — per-question review

Request: "Class 10 CBSE trigonometry, NCERT se, all question types, 2-3 assertion reason, 1 case study, 40 marks".
Provider: **anthropic** · latency 37.3 s · tokens in/out 7485/12929 · NCERT source: `offline_outline`.

**Objective 12/13 verified · Rubric 6/6 checked · 1 manual review · marks sum 40/40 ✔ · 2 verification round(s)**

| Q | type | marks | solved independently? | sympy | key matches? | status | notes |
|---|---|---|---|---|---|---|---|
| Q1 (q1) | mcq | 1 | yes → C | 1 option(s) equal check_expr | yes | verified | — |
| Q2 (q2) | mcq | 1 | yes → A | 1 option(s) equal check_expr | yes | verified | — |
| Q3 (q3) | mcq | 1 | yes → B | 1 option(s) equal check_expr | yes | verified | — |
| Q4 (q4) | mcq | 1 | yes → D | 1 option(s) equal check_expr | yes | verified | — |
| Q5 (q5) | mcq | 1 | yes → B | 1 option(s) equal check_expr | yes | verified | — |
| Q6 (q6) | mcq | 1 | yes → A | 1 option(s) equal check_expr | yes | verified | — |
| Q7 (q7) | mcq | 1 | yes → C | unparseable | — | manual_review | sympy could not parse options/check |
| Q8 (q8) | assertion_reason | 1 | yes → A | n/a (assertion-reason) | yes | verified | — |
| Q9 (q9) | assertion_reason | 1 | yes → D | n/a (assertion-reason) | yes | verified | — |
| Q10 (q10) | very_short | 2 | yes → 4/5 | key == check_expr | yes | verified | — |
| Q11 (q11) | very_short | 2 | yes → 2 | key == check_expr | yes | verified | — |
| Q12 (q12) | very_short | 2 | yes → 5/12 | key == check_expr | yes | verified | — |
| Q13 (q13) | very_short | 2 | yes → 17/13 | key == check_expr | yes | verified | — |
| Q14 (q14) | short_answer | 3 | no (rubric) | — | — | rubric_ok | — |
| Q15 (q15) | short_answer | 3 | no (rubric) | — | — | rubric_ok | — |
| Q16 (q16) | short_answer | 3 | no (rubric) | — | — | rubric_ok | — |
| Q17 (q17) | long_answer | 5 | no (rubric) | — | — | rubric_ok | — |
| Q18 (q18) | long_answer | 5 | no (rubric) | — | — | rubric_ok | — |
| Q19 (q19) | case_study | 4 | no (rubric) | — | — | rubric_ok | — |

## Errors found and fixed (3 runs)
- Run 1 (96 s, 3 rounds): q3/q4 "failed" were **verifier bugs**, not bad questions — `_norm` dropped the minus sign
  (`1` and `-1` looked identical) and `\dfrac` parsed as `d·frac`. q13 (`A = 45°, B = 15°`) → manual review (multi-value).
  Fixed: LaTeX → sympy converter (innermost-first `\sqrt`, `\[dt]frac`, `^\circ`, `\sin^2 x`), value-based option
  distinctness, pairwise `A=…, B=…` comparison. Regression tests in `tests/agent/test_core_security.py`.
- Run 2 (51 s): 5 manual reviews — `Rational(7,5)` check expressions rejected by an over-strict guard (bug), and A-R
  options (prose) were pushed through sympy (meaningless). Fixed: allow `Rational`, A-R decided by independent solver +
  standard option set. Re-checking run 2's saved paper then exposed a **real** error: an A-R item where the key said
  (B) and the independent solver said (A) → now `failed` → regenerated. Also a "prove that" in a 2-mark value slot →
  manual review; generator prompt tightened (single final value; Class 10 syllabus has no compound-angle formulae).
- Run 3 (this file): result above. The one manual review (q7) is the LaTeX shorthand `\\frac54` (no braces) —
  correctly NOT passed. Parser rule + regression test added after run 3 (`test_frac_shorthand_parses`); the result
  above is left exactly as the live run produced it.

Manual-review items are NOT counted as verified; a teacher must look at them before approval.
