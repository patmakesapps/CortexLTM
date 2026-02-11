import os
import socket
from urllib.parse import urlparse

from dotenv import load_dotenv
import psycopg2

load_dotenv()


class DatabaseUnavailableError(RuntimeError):
    """Raised when CortexLTM cannot establish a DB connection."""


def _clean_db_url(value: str | None) -> str | None:
    if not value:
        return None
    # Allow trailing inline comments in .env lines.
    raw = value.split(" #", 1)[0].strip()
    return raw or None


def _dsn_host_port(db_url: str) -> tuple[str | None, int]:
    parsed = urlparse(db_url)
    return parsed.hostname, parsed.port or 5432


def get_conn():
    db_url = _clean_db_url(os.getenv("SUPABASE_DB_URL"))
    if not db_url:
        raise DatabaseUnavailableError("Missing SUPABASE_DB_URL in .env")

    try:
        return psycopg2.connect(db_url)
    except psycopg2.OperationalError as exc:
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
