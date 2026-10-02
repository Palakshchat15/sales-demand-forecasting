"""Shared Postgres connection helper for the ML pipeline scripts."""

import os

import sqlalchemy


def get_engine():
    user = os.environ.get("POSTGRES_USER", "sales_user")
    password = os.environ.get("POSTGRES_PASSWORD", "sales_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "sales_forecast")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return sqlalchemy.create_engine(url)
