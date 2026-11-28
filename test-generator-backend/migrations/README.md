# Backend migrations

SQL here is applied by hand (Supabase SQL editor, as `postgres`); nothing runs automatically. Numbers share one
sequence with the frontend repo's `supabase/migrations/` (next free number: check both repos and all branches).
Every migration has a `_rollback.sql`.

## After migration 42: grant writes explicitly

Migration 42 changes the default privileges of schema `public`: a **new** table gives `anon` and `authenticated`
**SELECT only** (existing tables are not affected). So any new table the browser (or a user-JWT client) writes to
must say so in its own migration, next to its RLS policies:

```sql
alter table public.my_table enable row level security;
create policy my_table_insert_own on public.my_table for insert to authenticated with check (owner_id = auth.uid());
grant insert, update, delete on public.my_table to authenticated;   -- required since migration 42
```

Without the `grant`, the insert fails with `permission denied for table my_table`; it does not fail silently.
Do not grant writes to `anon`, and never grant `TRUNCATE`, `TRIGGER` or `REFERENCES` to either role.
