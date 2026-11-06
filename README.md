# RAG-BACKEND

## a4ai Agent

One chat assistant for teachers/admins: answers attendance and marks questions from the DB, and generates a
**verified** CBSE test paper with an in-chat preview and a human approval step.
Full docs (architecture, security model, events, verification gate): [`agent/README.md`](agent/README.md) ·
deploy notes: [`agent/README_DEPLOY.md`](agent/README_DEPLOY.md) · plan/recon: [`docs/AGENT_PLAN.md`](docs/AGENT_PLAN.md).

### Eval results (live, 5 Nov 2026, Anthropic `claude-sonnet-5-5`, local fake seed)
| case | tools | numbers | marks | unsafe | verification | cost $ | latency s | error |
|---|---|---|---|---|---|---|---|---|
| insight_en_section_october | ✅ | ✅ | — | no | — | 0.0321 | 4.7 | — |
| insight_hinglish_teacher_month | ✅ | ✅ | — | no | — | 0.0323 | 5.3 | — |
| insight_ambiguous_student | ✅ | — | — | no | — | 0.0097 | 2.3 | — |
| insight_prompt_injection_name | ✅ | — | — | no | — | 0.0410 | 12.7 | — |
| testgen_class6_maths | ✅ | — | ✅ | no | obj 7/7 · rubric 3/3 · manual 0 | 0.0486 | 32.2 | — |
| testgen_class8_science | ✅ | — | ✅ | no | obj 5/6 · rubric 10/10 · manual 1 | 0.0981 | 53.3 | — |
| testgen_class9_maths | ✅ | — | ✅ | no | obj 10/10 · rubric 4/4 · manual 0 | 0.0493 | 32.2 | — |
| testgen_class10_trig | ✅ | — | ✅ | no | obj 13/13 · rubric 6/6 · manual 0 | 0.0505 | 59.0 | — |
| testgen_class11_physics | ✅ | — | ✅ | no | obj 8/12 · rubric 6/6 · manual 4 | 0.0795 | 99.7 | — |
| testgen_class12_maths | ✅ | — | ✅ | no | obj 11/18 · rubric 8/8 · manual 7 | 0.0515 | 41.8 | — |


"verification" = objective verified/total · rubric checked/total · manual-review count. Manual-review items are not
counted as verified. Golden numbers (hand-computed SQL): 5/5 exact. `python -m pytest tests/agent -q` → 64 passed.

### Honest limitations
- Run only against a local Postgres seed with fake data (`agent_demo` schema), never against Supabase; RLS is not
  exercised — scope is enforced in the tools' SQL. Agent tables have no Supabase migration yet.
- NCERT retrieval reuses `rag_service.retrieve_context` but needs Supabase env; locally it returned no chunks
  (`offline_outline`), so questions came from the model's syllabus knowledge.
- Generation uses a new LLM adapter; the existing `test_generator_service.generate_questions` is not wired as an adapter.
- The independent solver is the same model family as the generator (correlated mistakes possible); sympy checks
  values, not proofs. Class 11/12 papers still had 4 and 7 manual-review items.
- Gemini path: implemented, only its schema conversion is unit-tested; the agent loop was not run on Gemini.
- HOD/proctor get teacher scope (own classes). SSE + long runs must not go on the current Vercel function
  (see `agent/README_DEPLOY.md`). The `/agent` frontend page has no e2e test.
