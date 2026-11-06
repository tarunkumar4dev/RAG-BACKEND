# Schema-faithful fixture for the real-data agent tests

Vendored copies (do not edit by hand; re-copy when the source changes):
- `00_supabase_stub.sql`, `01_baseline_from_audit.sql` ← a4ai/supabase/tests/upload_flow/ (baseline generated from the
  live schema audit, 29 Sep 2026)
- `m_*.sql` ← a4ai/supabase/migrations/ (01, 02, 05, 06a, 10, 11, 12) — the migrations that add the columns the agent reads
- `02_tests_table_from_supabase_migration.sql` ← RAG-BACKEND/test-generator-backend/supabase_migration.sql (`tests`)
Load order = `LOAD_ORDER` in `tests/agent/real_fixture/build.py`. Data: `../data.sql` (fake, hand-computed values).
