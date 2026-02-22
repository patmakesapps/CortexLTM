import os
import socket
import threading
from urllib.parse import urlparse

from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()


class DatabaseUnavailableError(RuntimeError):
    """Raised when CortexLTM cannot establish a DB connection."""


class _PooledConnection:
    """Proxy psycopg2 connection that returns to pool on close()."""

    def __init__(self, pool: ThreadedConnectionPool, conn: psycopg2.extensions.connection):
        self._pool = pool
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def close(self) -> None:
        if self._conn is None:
            return
        self._pool.putconn(self._conn)
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


_pool_lock = threading.Lock()
_pool: ThreadedConnectionPool | None = None
_pool_db_url: str | None = None


def _clean_db_url(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.split(" #", 1)[0].strip()
    return raw or None


def _dsn_host_port(db_url: str) -> tuple[str | None, int]:
    parsed = urlparse(db_url)
    return parsed.hostname, parsed.port or 5432


def _raise_db_unavailable(db_url: str, exc: Exception) -> None:
    host, port = _dsn_host_port(db_url)
    if host:
        try:
            socket.getaddrinfo(host, port)
        except socket.gaierror as dns_exc:
            raise DatabaseUnavailableError(
                f"Cannot resolve database host '{host}'. "
                "Verify SUPABASE_DB_URL project ref/host in Supabase dashboard "
                "or local DNS/network policy."
            ) from dns_exc
    raise DatabaseUnavailableError(f"Failed to connect to database: {exc}") from exc


def _pool_limits() -> tuple[int, int]:
    minconn_raw = os.getenv("CORTEX_DB_POOL_MIN", "1").strip()
    maxconn_raw = os.getenv("CORTEX_DB_POOL_MAX", "12").strip()
    try:
        minconn = max(1, int(minconn_raw))
    except Exception:
        minconn = 1
    try:
        maxconn = max(minconn, int(maxconn_raw))
    except Exception:
        maxconn = max(4, minconn)
    return minconn, maxconn


def _get_pool(db_url: str) -> ThreadedConnectionPool:
    global _pool, _pool_db_url
    with _pool_lock:
        if _pool is not None and _pool_db_url == db_url:
            return _pool

        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:
                pass
            _pool = None
            _pool_db_url = None

        minconn, maxconn = _pool_limits()
        try:
            _pool = ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, dsn=db_url)
        except Exception as exc:
            _raise_db_unavailable(db_url, exc)

        _pool_db_url = db_url
        return _pool


def get_conn():
    db_url = _clean_db_url(os.getenv("SUPABASE_DB_URL"))
    if not db_url:
        raise DatabaseUnavailableError("Missing SUPABASE_DB_URL in .env")

    pool = _get_pool(db_url)
    try:
        conn = pool.getconn()
    except Exception as exc:
        _raise_db_unavailable(db_url, exc)
    return _PooledConnection(pool, conn)
