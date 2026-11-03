import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def connect():
    url = os.environ.get('DATABASE_URL')
    if url:
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(url, row_factory=dict_row, connect_timeout=10)
    if os.environ.get('DATABASE_MODE') == 'sqlite':
        conn = sqlite3.connect(os.environ.get('SQLITE_PATH', str(ROOT / 'local.db')))
        conn.row_factory = sqlite3.Row
        return conn
    raise RuntimeError('Set DATABASE_URL for PostgreSQL or DATABASE_MODE=sqlite for local practice.')


def execute(conn, sql, params=()):
    if isinstance(conn, sqlite3.Connection):
        return conn.execute(sql.replace('%s', '?'), params)
    return conn.execute(sql, params)


def initialize(conn):
    sql = (ROOT / 'schema.sql').read_text()
    if isinstance(conn, sqlite3.Connection):
        conn.executescript(sql)
    else:
        conn.execute(sql)
    conn.commit()
