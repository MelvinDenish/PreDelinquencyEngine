"""
Centralized PostgreSQL connection factory for PDI Engine.
All connections go through here — ensures SSL is applied consistently,
avoids 9+ repeated psycopg2.connect() blocks scattered across the codebase.
"""
import logging
import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from config.settings import PostgresConfig

logger = logging.getLogger(__name__)


def get_pg_connection(timeout: int = 5):
    """
    Return a psycopg2 connection with SSL enforced if configured.
    Always close the connection after use — prefer using get_pg_cursor() context manager.
    """
    ssl_mode = os.getenv("POSTGRES_SSLMODE", "prefer")  # 'require' in production
    ssl_root_cert = os.getenv("POSTGRES_SSLROOTCERT")

    connect_kwargs = dict(
        host=PostgresConfig.HOST,
        port=PostgresConfig.PORT,
        user=PostgresConfig.USER,
        password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
        connect_timeout=timeout,
        sslmode=ssl_mode,
    )
    if ssl_root_cert and os.path.exists(ssl_root_cert):
        connect_kwargs["sslrootcert"] = ssl_root_cert

    return psycopg2.connect(**connect_kwargs)


@contextmanager
def get_pg_cursor(dict_cursor: bool = False, autocommit: bool = False):
    """
    Context manager that provides a cursor and handles commit/rollback/close.

    Usage:
        with get_pg_cursor() as cursor:
            cursor.execute("SELECT ...")
            rows = cursor.fetchall()
    """
    conn = get_pg_connection()
    if autocommit:
        conn.autocommit = True
    try:
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = conn.cursor(cursor_factory=cursor_factory) if cursor_factory else conn.cursor()
        try:
            yield cursor
            if not autocommit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    finally:
        conn.close()


def execute_query(sql: str, params: tuple = (), fetch: str = "all", dict_cursor: bool = False):
    """
    Execute a single query and return results.
    fetch: 'all', 'one', or 'none'
    """
    with get_pg_cursor(dict_cursor=dict_cursor) as cursor:
        cursor.execute(sql, params)
        if fetch == "all":
            return cursor.fetchall()
        elif fetch == "one":
            return cursor.fetchone()
        return None


def execute_write(sql: str, params: tuple = ()) -> int:
    """Execute an INSERT/UPDATE/DELETE and return rowcount."""
    with get_pg_cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.rowcount
