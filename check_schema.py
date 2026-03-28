import psycopg2
conn = psycopg2.connect(host='pdi-postgres', port=5432, user='pdi_user', password='pdi_password', dbname='pdi_db')
cur = conn.cursor()

# Check all tables
cur.execute("""
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' ORDER BY table_name
""")
print("=== TABLES ===")
for r in cur.fetchall():
    print(f"  {r[0]}")

# Check transactions columns
cur.execute("""
    SELECT column_name, data_type FROM information_schema.columns 
    WHERE table_name = 'transactions' ORDER BY ordinal_position
""")
print("\n=== transactions columns ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

# Check payment_events columns
cur.execute("""
    SELECT column_name, data_type FROM information_schema.columns 
    WHERE table_name = 'payment_events' ORDER BY ordinal_position
""")
print("\n=== payment_events columns ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

# Check account_balances columns
cur.execute("""
    SELECT column_name, data_type FROM information_schema.columns 
    WHERE table_name = 'account_balances' ORDER BY ordinal_position
""")
print("\n=== account_balances columns ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

# Check customers columns
cur.execute("""
    SELECT column_name, data_type FROM information_schema.columns 
    WHERE table_name = 'customers' ORDER BY ordinal_position
""")
print("\n=== customers columns ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

cur.close()
conn.close()
