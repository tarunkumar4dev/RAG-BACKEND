"""Gunicorn config for the a4ai Agent (production).

uvicorn workers handle async lifespan + sync endpoints via anyio thread pool.
WEB_CONCURRENCY controls worker count (default 2 for the trial).
"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300  # paper generation can take 4+ minutes
graceful_timeout = 30
keepalive = 5

# Logging: let agent.logging_config handle structured JSON; gunicorn uses its default
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Preload so the connection pool and model routing are shared
preload_app = True

# Limits
max_requests = 1000  # recycle workers after 1k requests (leak guard)
max_requests_jitter = 100
