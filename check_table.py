"""One-off check that the public tables (and ncert_chunks) exist.

Credentials come only from the environment, never from this file:
    DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/postgres?sslmode=require python check_table.py
"""
import os
import sys

import psycopg2

dsn = os.environ.get("DATABASE_URL", "").strip()
if not dsn:
    sys.exit("DATABASE_URL is not set (see the docstring).")

conn = psycopg2.connect(dsn, sslmode="require")
conn.set_session(readonly=True)

cursor = conn.cursor()

# Check all tables
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
tables = cursor.fetchall()
print("All tables:", [t[0] for t in tables])

# Check specifically for ncert_chunks
cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='ncert_chunks')")
exists = cursor.fetchone()[0]
print(f"ncert_chunks table exists: {exists}")

conn.close()
