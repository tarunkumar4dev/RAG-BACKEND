"""Side-effect-free latency probe for rag-backend (no Gemini, no payments, no writes)."""
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = "https://rag-backend-kappa-three.vercel.app"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def b64(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")


# Well-formed but invalid token: passes the local checks, then Supabase Auth rejects it (401)
FAKE = f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64({'sub': '00000000-0000-0000-0000-000000000000', 'exp': int(time.time()) + 3600})}.x"

PROBES = [
    ("health (no DB)", "GET", "/health", None),
    ("chapters (DB read)", "GET", "/api/v1/test-generator/chapters?subject=Science&class_grade=10", None),
    ("checker 401 (Auth round trip)", "POST", "/api/v1/copy-checker/sheets/00000000-0000-0000-0000-000000000000/check", FAKE),
]


def once(method, path, token):
    req = urllib.request.Request(BASE + path, method=method, data=b"{}" if method == "POST" else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        e.read()
        code = e.code
    return (time.perf_counter() - t) * 1000, code


for name, method, path, token in PROBES:
    once(method, path, token)                       # warm-up (cold start excluded)
    runs = [once(method, path, token) for _ in range(N)]
    ms = [r[0] for r in runs]
    print(f"{name:32s} median={statistics.median(ms):6.0f} ms  min={min(ms):5.0f}  max={max(ms):5.0f}  codes={sorted({r[1] for r in runs})}")
