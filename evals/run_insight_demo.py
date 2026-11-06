"""Admin demo: 'Sharma ma'am ki is mahine ki attendance kya hai?' -> event log (JSONL)."""
import json, os, sys, uuid
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core import Agent
from agent.db import connect
from agent.llm import get_provider
from agent.schemas import SessionCtx

ADMIN = "aaaaaaaa-0000-0000-0000-000000000001"
MAIN = "11111111-1111-1111-1111-111111111111"
msg = sys.argv[2] if len(sys.argv) > 2 else "Sharma ma'am ki is mahine ki attendance kya hai?"
ctx = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
conn = connect(readonly=False)
sid = str(uuid.uuid4())
conn.execute("insert into agent_sessions(id,user_id,institute_id,role) values (%s,%s,%s,%s)", (sid, ADMIN, MAIN, "admin"))
agent = Agent(get_provider(sys.argv[1] if len(sys.argv) > 1 else "anthropic"), ctx, sid, conn_rw=conn)
out = os.path.join(os.path.dirname(__file__), "sample_sharma_attendance_events.jsonl")
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps({"user_message": msg, "role": "admin"}, ensure_ascii=False) + "\n")
    for ev in agent.run(msg):
        f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        print(ev["type"], json.dumps({k: v for k, v in ev.items() if k not in ("seq", "request_id", "type")},
                                     ensure_ascii=False, default=str)[:300])
