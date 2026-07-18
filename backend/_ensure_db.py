"""Ensure Postgres database trust_simulator exists."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("trust_simulator",))
if cur.fetchone():
    print("DB trust_simulator already exists")
else:
    cur.execute("CREATE DATABASE trust_simulator")
    print("Created DB trust_simulator")
cur.close()
conn.close()
