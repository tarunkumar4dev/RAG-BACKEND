"""
a4ai NCERT Test Builder load test (zero-cost: read-only, no Gemini / paid APIs).

User classes (pick one on the CLI):
  TeacherUser    realistic drag-and-drop Test Builder session (edge cache allowed)
  CacheBustUser  small scenario: random `_lt` param skips Vercel's edge cache so every
                 request reaches the Python function (it still hits the in-memory cache,
                 whose key ignores unknown params)

Env vars:
  LOADTEST_BYPASS_TOKEN     sent as X-Loadtest-Token; the backend honours it only when
                            VERCEL_ENV == "preview" (see app/main.py)
  VERCEL_PROTECTION_BYPASS  sent as x-vercel-protection-bypass, to get past Vercel
                            Deployment Protection on the preview URL
"""
import json
import os
import random
import re
import uuid
from collections import Counter
from urllib.parse import urlsplit

import gevent
from locust import HttpUser, between, events, task


def _load_local_env():
    """Read secrets from loadtest/.env.loadtest (gitignored) without ever printing them."""
    path = os.path.join(os.path.dirname(__file__), ".env.loadtest")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()

API = "/api/v1/test-generator"
STATS_PATH = f"{API}/ncert-question-stats"
QUESTIONS_PATH = f"{API}/ncert-questions"

# SAFETY: only these paths may ever be requested. Everything else (generate, export,
# modules, chat, checkers, whatsapp, payment, ...) uses Gemini or writes data.
ALLOWED_PATHS = {STATS_PATH, QUESTIONS_PATH}
BLOCKED_PREFIXES = (
    f"{API}/generate", f"{API}/verify-answers", f"{API}/feedback", f"{API}/export",
    f"{API}/save", f"{API}/tests/", f"{API}/quiz/", "/api/v1/modules", "/modules",
    "/api/v1/worksheet", "/worksheet", "/api/v1/chat", "/api/v1/whatsapp",
    "/api/v1/payment", "/api/v1/contest", "/api/v1/community", "/api/v1/test-checker",
    "/api/v1/copy-checker", "/api/v1/smart-assignment",
)
# Host allowlist: only a Vercel *branch* preview alias (never git-main), or a local mock.
# Production has several aliases (kappa-three, rag-backend-<team>, git-main, api.a4ai.in) plus a
# unique URL per deploy, so a blocklist can't cover it; an allowlist can.
PREVIEW_HOST = re.compile(r"^rag-backend-git-(?!main-)[a-z0-9-]+-taruns-projects-5d2bf9e6\.vercel\.app$")
LOCAL_HOSTS = {"127.0.0.1", "localhost"}

PAGE_SIZE = 30  # same as TestBuilderPage.tsx

# class -> subjects, from TestBuilderPage SUBJECTS_BY_CLASS intersected with what
# ncert_questions actually holds (checked 2026-10-20; class 8 has no rows, so it is left out).
COMBOS = [
    ("6", "Science"), ("6", "Mathematics"),
    ("7", "Mathematics"),
    ("9", "Science"), ("9", "Mathematics"), ("9", "English"), ("9", "Economics"),
    ("9", "Geography"), ("9", "History"), ("9", "Political Science"),
    ("10", "Science"), ("10", "Mathematics"), ("10", "English"), ("10", "Economics"),
    ("10", "Geography"), ("10", "History"), ("10", "Political Science"),
    ("11", "Accountancy"), ("11", "Physics"), ("11", "Chemistry"), ("11", "Biology"),
    ("11", "Mathematics"), ("11", "Economics"), ("11", "History"), ("11", "Political Science"),
    ("12", "Accountancy"), ("12", "Physics"), ("12", "Chemistry"), ("12", "Biology"),
    ("12", "Mathematics"), ("12", "Economics"), ("12", "English"), ("12", "History"),
    ("12", "Political Science"),
]
QUESTION_TYPES = ["exercise", "example", "intext"]

vercel_cache = Counter()
first_requests = []  # first 5 (name, status, x-vercel-cache), printed live and saved
cpu_samples = []  # Locust's own CPU %, sampled every 5 s; > 90% means the generator is the bottleneck
CPU_LIMIT = 90
_env = None
_hard_stopped = False


def _abort_reason(response):
    """Hard-stop conditions: any of these means the test setup is broken, not the backend."""
    status = response.status_code
    headers = response.headers or {}
    if status == 0:
        return None  # connection error / client timeout: a load result, counted as a failure
    if status == 401 or 300 <= status < 400:
        return f"HTTP {status}: Vercel protection bypass not working"
    if status == 429:
        return "HTTP 429: rate-limit bypass is not working"
    if status == 403:
        return "HTTP 403: blocked by Vercel (protection/firewall) or bypass secret wrong"
    if headers.get("x-vercel-mitigated"):
        return f"Vercel mitigation: x-vercel-mitigated={headers['x-vercel-mitigated']}"
    if "text/html" in headers.get("content-type", "") and "Vercel Security Checkpoint" in (response.text or ""):
        return "Vercel challenge page (Security Checkpoint)"
    # 5xx (e.g. Vercel's plain-text 504 timeout) is a real load result: counted as a failure, not a stop.
    if status < 500 and "application/json" not in headers.get("content-type", ""):
        return f"HTTP {status}: non-JSON response (content-type={headers.get('content-type')!r})"
    return None


def _hard_stop(reason):
    global _hard_stopped
    if _hard_stopped:
        return
    _hard_stopped = True
    print(f"HARD STOP: {reason}")
    if _env and _env.runner:
        _env.process_exit_code = 3
        gevent.spawn(_env.runner.quit)  # not from this user's greenlet, which quit() kills


def _check_path(url: str) -> None:
    path = urlsplit(url).path
    if path.startswith(BLOCKED_PREFIXES) or path not in ALLOWED_PATHS:
        raise RuntimeError(f"BLOCKED: load test tried to call {path!r}")


@events.test_start.add_listener
def _check_host(environment, **_):
    host = urlsplit(environment.host or "").hostname or ""
    if host not in LOCAL_HOSTS and not PREVIEW_HOST.match(host):
        raise SystemExit(f"Refusing to load test {host!r}: only rag-backend-git-<branch>-... "
                         "preview aliases (not main) or localhost are allowed.")
    if not os.getenv("LOADTEST_BYPASS_TOKEN"):
        print("WARNING: LOADTEST_BYPASS_TOKEN not set; expect 429s from the rate limiter.")
    gevent.spawn(_sample_cpu, environment)


def _sample_cpu(environment):
    gevent.sleep(5)  # skip the import/spawn spike
    while environment.runner and environment.runner.state not in ("stopping", "stopped", "quitting"):
        cpu_samples.append(environment.runner.current_cpu_usage)
        gevent.sleep(5)


def _cpu_summary():
    if not cpu_samples:
        return {"avg": None, "max": None, "generator_bound": False}
    peak = max(cpu_samples)
    return {"avg": round(sum(cpu_samples) / len(cpu_samples), 1), "max": round(peak, 1),
            "generator_bound": peak > CPU_LIMIT}


@events.init.add_listener
def _keep_env(environment, **_):
    global _env
    _env = environment


@events.request.add_listener
def _on_request(name=None, response=None, **_):
    if response is None or getattr(response, "headers", None) is None:
        return
    cache = response.headers.get("x-vercel-cache", "NONE")
    vercel_cache[cache] += 1
    if len(first_requests) < 5:
        first_requests.append({"name": name, "status": response.status_code, "x-vercel-cache": cache})
        print(f"[first {len(first_requests)}/5] {response.status_code} x-vercel-cache={cache} {name}")
    reason = _abort_reason(response)
    if reason:
        _hard_stop(reason)


@events.test_stop.add_listener
def _report_cache(environment, **_):
    total = sum(vercel_cache.values()) or 1
    summary = {k: f"{v} ({v * 100 / total:.0f}%)" for k, v in vercel_cache.most_common()}
    print(f"x-vercel-cache: {json.dumps(summary)}")
    cpu = _cpu_summary()
    print(f"locust CPU %: avg={cpu['avg']} max={cpu['max']}"
          + ("  ** GENERATOR-BOUND, RESULT UNRELIABLE **" if cpu["generator_bound"] else ""))
    prefix = environment.parsed_options.csv_prefix if environment.parsed_options else None
    if prefix:
        with open(f"{prefix}_vercel_cache.json", "w", encoding="utf-8") as f:
            json.dump({"cache": dict(vercel_cache), "first_requests": first_requests,
                       "hard_stopped": _hard_stopped, "locust_cpu": cpu}, f, indent=2)


class _BuilderUser(HttpUser):
    abstract = True
    wait_time = between(2, 6)
    cache_bust = False

    def on_start(self):
        # Guard every request this user makes, including any added later by mistake.
        original = self.client.request
        # Secret headers are attached per request, only after the allowlist check passes,
        # so they can never ride along on a blocked path.
        secret_headers = {
            name: value for name, value in (
                ("X-Loadtest-Token", os.getenv("LOADTEST_BYPASS_TOKEN")),
                ("x-vercel-protection-bypass", os.getenv("VERCEL_PROTECTION_BYPASS")),
            ) if value
        }

        def guarded(method, url, *args, **kwargs):
            if method.upper() != "GET":
                raise RuntimeError(f"BLOCKED: only GET allowed, got {method} {url}")
            _check_path(url)
            kwargs["headers"] = {**(kwargs.get("headers") or {}), **secret_headers}
            kwargs["allow_redirects"] = False  # a redirect means protection bypass failed; see it

            return original(method, url, *args, **kwargs)

        self.client.request = guarded
        self.chapters = []
        self.pick_combo()

    def _get_json(self, path, params, name):
        if self.cache_bust:
            params = {**params, "_lt": uuid.uuid4().hex}
            name += " [bust]"
        with self.client.get(path, params=params, name=name, catch_response=True) as resp:
            if resp.status_code == 0:
                resp.failure(f"connection error: {type(resp.error).__name__}")
                return None
            if resp.status_code >= 500:
                # Body snippet tells a Vercel 504 timeout apart from a FastAPI 500. Vercel request
                # ids (bom1::...) are masked so identical errors group into one failures.csv row.
                snippet = re.sub(r"\S+::\S+", "<vercel-id>", " ".join((resp.text or "").split()))[:100]
                resp.failure(f"HTTP {resp.status_code}: {snippet}")
                return None
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return None
            try:
                data = resp.json()
            except ValueError:
                resp.failure("non-JSON body (Vercel error page?)")
                _hard_stop(_abort_reason(resp) or "HTTP 200: JSON parse failed")
                return None
            if not data.get("ok"):
                # Backend reports DB errors as 200 + ok:false.
                resp.failure("ok:false")
                return None
            resp.success()
            return data

    def pick_combo(self):
        self.class_grade, self.subject = random.choice(COMBOS)
        data = self._get_json(
            STATS_PATH, {"subject": self.subject, "class_grade": self.class_grade},
            name="1 stats [class/subject]",
        )
        self.chapters = [c for c in (data or {}).get("chapters", []) if c.get("chapter")]

    def questions(self, chapter, offset=0, question_type=None, name="3 questions page1"):
        params = {
            "subject": self.subject, "class_grade": self.class_grade, "chapter": chapter,
            "limit": PAGE_SIZE, "offset": offset,
        }
        if question_type:
            params["question_type"] = question_type
        return self._get_json(QUESTIONS_PATH, params, name=name)


class TeacherUser(_BuilderUser):
    """Mirrors TestBuilderPage: stats -> prefetch first 3 chapters -> open/paginate/filter."""

    @task(1)
    def switch_class_subject(self):
        self.pick_combo()
        for ch in self.chapters[:3]:  # page prefetches first 3 chapters 300ms after stats
            self.questions(ch["chapter"], name="2 prefetch chapter")

    @task(5)
    def browse_chapter(self):
        if not self.chapters:
            self.pick_combo()
            if not self.chapters:
                return
        stat = random.choice(self.chapters)
        data = self.questions(stat["chapter"])
        pages = 1
        while data and data.get("hasMore") and pages < 3 and random.random() < 0.5:
            self.wait()
            data = self.questions(stat["chapter"], offset=pages * PAGE_SIZE, name="4 questions next page")
            pages += 1

    @task(2)
    def filter_by_type(self):
        if not self.chapters:
            return
        stat = random.choice(self.chapters)
        types = [t["name"] for t in stat.get("types", []) if t.get("name") in QUESTION_TYPES]
        self.questions(stat["chapter"], question_type=random.choice(types or QUESTION_TYPES),
                       name="5 questions filter type")


class CacheBustUser(_BuilderUser):
    """Every request has a unique `_lt` param: measures the Python function, not Vercel's CDN."""
    cache_bust = True

    @task(1)
    def stats(self):
        self.pick_combo()

    @task(3)
    def chapter(self):
        if not self.chapters:
            return
        self.questions(random.choice(self.chapters)["chapter"])
