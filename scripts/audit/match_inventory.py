import json, re, subprocess, sys, glob, os
sys.stdout.reconfigure(encoding="utf-8")
inv, repos = sys.argv[1], sys.argv[2:]
d = json.load(open(inv, encoding="utf-8")); rows = d["rows"] if isinstance(d, dict) else d
parts = []
for r in repos:
    out = subprocess.run(["git", "-C", r, "log", "--all", "-p", "--no-color", "--format=", "--", "*.sql", "*.py", "*.ts", "*.tsx", "*.md", "*.json", "*.txt"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    parts.append("\n".join(l[1:] for l in out.splitlines() if l.startswith("+")))
    for f in glob.glob(os.path.join(r, "**", "*.sql"), recursive=True):   # untracked local SQL too
        if "node_modules" not in f: parts.append(open(f, encoding="utf-8", errors="replace").read())
corpus = "\n".join(parts).lower()
print(f"corpus: {len(corpus)/1e6:.1f} MB of DDL/code history")
DEFAULT = {
 "extension": {"plpgsql", "pgcrypto", "uuid-ossp", "pg_stat_statements", "pg_graphql", "supabase_vault", "pgjwt", "pgsodium"},
 "event_trigger": {"issue_graphql_placeholder", "issue_pg_cron_access", "issue_pg_graphql_access", "issue_pg_net_access", "pgrst_ddl_watch", "pgrst_drop_watch"},
 "publication": {"supabase_realtime", "supabase_realtime_messages_publication"},
}
def q(name): return r'"?' + re.escape(name.lower()) + r'"?'
def matched(r):
    k, s, n = r["kind"], (r["schema_"] or "").lower(), r["name"]
    if n.lower() in DEFAULT.get(k, set()): return "supabase default"
    sq = rf'(?:{q(s)}\.)?' if s else ''
    pats = {
      "table": [rf'(create|alter)\s+(unlogged\s+)?(table|view|materialized\s+view)\s+(if\s+not\s+exists\s+)?(or\s+replace\s+)?{sq}{q(n)}\b', rf'create\s+(or\s+replace\s+)?(materialized\s+)?view\s+{sq}{q(n)}\b', rf'create\s+table\s+{q(s)}\.{q(n)}\s+as'],
      "function": [rf'function\s+{sq}{q(n)}\s*\('],
      "trigger": [rf'create\s+(or\s+replace\s+)?(constraint\s+)?trigger\s+{q(n)}\b'],
      "trigger(system schema)": [rf'create\s+(or\s+replace\s+)?trigger\s+{q(n)}\b'],
      "role": [rf'create\s+(role|user)\s+{q(n)}\b'],
      "extension": [rf'create\s+extension\s+(if\s+not\s+exists\s+)?{q(n)}'],
      "schema": [rf'create\s+schema\s+(if\s+not\s+exists\s+)?{q(n)}\b'],
      "publication": [rf'publication\s+{q(n)}'],
      "cron_job": [rf"cron\.schedule\s*\(\s*'{re.escape(n.lower())}'"],
      "storage_bucket": [rf"['\"]{re.escape(n.lower())}['\"]"],
      "vault_secret": [rf"['\"]{re.escape(n.lower())}['\"]"],
      "event_trigger": [rf'event\s+trigger\s+{q(n)}'],
    }.get(k, [])
    for p in pats:
        if re.search(p, corpus): return "in repo"
    return None
un = [r for r in rows if not matched(r)]
print(f"objects: {len(rows)}, unmatched: {len(un)}")
for r in un: print(f"  {r['kind']:<22} {r['schema_']}.{r['name']}  {r['detail'][:150]}")
print("\n== weak check: unmatched objects whose name is not referenced anywhere in either repo's history")
tabs = []
for r in un:
    n = r["name"].lower()
    if r["kind"].startswith("trigger(system") and r["schema_"] == "storage": continue
    hits = len(re.findall(rf"\b{re.escape(n)}\b", corpus))
    if r["kind"] == "table": tabs.append((r["schema_"], r["name"]))
    if hits == 0: print(f"  NOT REFERENCED  {r['kind']:<10} {r['schema_']}.{r['name']}  {r['detail'][:120]}")
open(sys.argv[1] + ".tables", "w").write("\n".join(f"{s}.{n}" for s, n in tabs))
