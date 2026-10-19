"""
app/core/db_pool.py — Thread-safe PostgreSQL connection pooling with proxy close
"""
import os
import logging
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_pool = None

# Direct Postgres is used only by Modules, Worksheet and the NCERT question stats.
# Credentials come ONLY from the environment (this repo is public): DATABASE_URL, or DB_PASSWORD (+ optional
# DB_HOST / DB_NAME / DB_USER / DB_PORT, whose defaults are not secret).
MISSING_CONFIG_MSG = ("Direct DB is not configured: set DATABASE_URL (or DB_PASSWORD) in the environment. "
                      "Modules, Worksheet and NCERT question stats will fail until it is set.")


def config_error():
    """None when credentials are configured, else the reason (logged at startup by app.main)."""
    if os.getenv("DATABASE_URL") or os.getenv("DB_PASSWORD"):
        return None
    return MISSING_CONFIG_MSG


def _connect_kwargs() -> dict:
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return {"dsn": db_url, "sslmode": "require", "connect_timeout": 10}
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError(MISSING_CONFIG_MSG)
    return {
        "host": os.getenv("DB_HOST", "aws-0-ap-south-1.pooler.supabase.com"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres.dcmnzvjftmdbywrjkust"),
        "password": password,
        "port": int(os.getenv("DB_PORT", "6543")),
        "sslmode": "require",
        "connect_timeout": 10,
    }


def _get_pool():
    global _pool
    if _pool is None or _pool.closed:
        try:
            _pool = ThreadedConnectionPool(minconn=1, maxconn=10, **_connect_kwargs())
            logger.info("ThreadedConnectionPool initialized (1-10 connections)")
        except Exception as e:
            logger.error(f"Failed to initialize ThreadedConnectionPool: {e}")
            _pool = None
    return _pool


class PooledConnectionProxy:
    """Wraps a pooled connection so calling .close() returns it to the pool instead of terminating it."""
    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        self._closed = False

    def close(self):
        if not self._closed and self._pool and self._conn:
            self._closed = True
            try:
                if not self._conn.closed:
                    self._conn.rollback()
                self._pool.putconn(self._conn)
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")
                try:
                    self._pool.putconn(self._conn, close=True)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_db_connection():
    """Returns a PooledConnectionProxy whose .close() returns connection to pool."""
    pool = _get_pool()
    if not pool:
        try:
            conn = psycopg2.connect(**_connect_kwargs())
            conn.autocommit = True
            return conn
        except Exception as e:
            logger.error(f"Direct connection failed: {e}")
            return None

    try:
        conn = pool.getconn()
        if conn.closed != 0:
            pool.putconn(conn, close=True)
            conn = pool.getconn()
        conn.autocommit = True
        return PooledConnectionProxy(pool, conn)
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        return None

