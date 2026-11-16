FROM python:3.12-slim AS base

RUN groupadd -r agent && useradd -r -g agent -s /sbin/nologin agent

WORKDIR /app

COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt gunicorn

COPY agent/ /app/agent/
COPY scripts/export_openapi.py /app/scripts/export_openapi.py
COPY docs/agent_openapi_v2.json /app/docs/agent_openapi_v2.json

RUN mkdir -p /app/agent/.devdata/files && chown -R agent:agent /app/agent/.devdata

USER agent

ENV PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    AGENT_LOG_FORMAT=json \
    AGENT_CORS_ORIGINS="" \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["gunicorn", "agent.api:app", "-c", "agent/gunicorn.conf.py"]
