"""Postgres connection helper for the inference database (Phase 16/17).

Shared by src/femto_rul/serving/telemetry.py (writes predictions) and
src/femto_rul/monitoring/current.py (reads them back for drift reports),
so the connection contract only lives in one place.
"""

from __future__ import annotations

import psycopg2

from femto_rul import config


def get_connection(dbname: str = config.INFERENCE_DB_NAME):
    """Open a new connection to a Postgres database in this project's stack.

    Caller owns the connection lifecycle (close it, or use it as a context
    manager). Host/port default to the docker-compose service name/port —
    override via INFERENCE_DB_HOST/INFERENCE_DB_PORT for local, non-compose
    development.
    """
    return psycopg2.connect(
        host=config.INFERENCE_DB_HOST,
        port=config.INFERENCE_DB_PORT,
        dbname=dbname,
        user=config.POSTGRES_USER,
        password=config.require_env("POSTGRES_PASSWORD"),
    )
