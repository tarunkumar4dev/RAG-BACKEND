import psycopg2

conn = psycopg2.connect(
    host="db.dcmnzvjftmdbywrjkust.supabase.co",
    database="postgres",
    user="postgres",
    password="YOUR_NEW_PASSWORD",
    port="5432",
    sslmode="require"
)

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