"""Create the Education Beast contest from contest_payload.json via the existing POST /contests.

Usage (from the folder containing both files):
  python post_contest.py contest_payload.json --dry-run          # validate + print payload summary, no network
  JWT=<teacher access token> API=https://<backend-host>/api/v1 python post_contest.py contest_payload.json
Optional env:
  OPENS_AT / CLOSES_AT   ISO with offset, e.g. 2026-10-25T16:00:00+05:30 (enforced only once the new backend is deployed)
  DURATION               minutes, 5-180 (default from JSON: 60)
  MAX_WARNINGS           tab-switch limit before auto-submit (default 5)
  TITLE                  override title
"""
import os, sys, json, urllib.request, urllib.error

if len(sys.argv) < 2:
    sys.exit(__doc__)
p = json.load(open(sys.argv[1], encoding="utf-8-sig"))   # -sig: the payload may carry a UTF-8 BOM (Windows editors)
dry = "--dry-run" in sys.argv

qs = p["questions"]
problems = []
for i, q in enumerate(qs, 1):
    labels = [o["label"] for o in q["options"]]
    if q["correct_answer"] not in labels: problems.append(f"Q{i}: answer {q['correct_answer']} not in {labels}")
    if len(labels) < 2: problems.append(f"Q{i}: <2 options")
    if not q["question_text"].strip(): problems.append(f"Q{i}: empty text")
if problems:
    sys.exit("Fix before posting:\n" + "\n".join(problems))

body = {
    **p,
    "title": os.environ.get("TITLE", p["title"]),
    "duration_minutes": min(max(int(os.environ.get("DURATION", p.get("duration_minutes") or 60)), 5), 180),
    "max_attempts": 1,
    "answer_mode": "after_test",
    "show_explanation": True,
    "enable_camera": True,
    "enable_tab_detection": True,
    "allow_back_navigation": True,
    "max_warnings": int(os.environ.get("MAX_WARNINGS", 5)),
}
if os.environ.get("OPENS_AT"): body["scheduled_at"] = os.environ["OPENS_AT"]
if os.environ.get("CLOSES_AT"): body["expires_at"] = os.environ["CLOSES_AT"]

print(f"{len(qs)} questions | {sum(q['marks'] for q in qs)} marks | {body['duration_minutes']} min | "
      f"max_warnings={body['max_warnings']} | camera={body['enable_camera']} | key={''.join(q['correct_answer'] for q in qs)}")
if dry:
    sys.exit(0)

API, JWT = os.environ["API"].rstrip("/"), os.environ["JWT"]
req = urllib.request.Request(API + "/contests", method="POST", data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + JWT, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print(json.dumps(json.load(r), indent=2))   # contest_id, short_code, share_link
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()}")