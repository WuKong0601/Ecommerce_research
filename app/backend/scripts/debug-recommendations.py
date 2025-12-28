"""
Debug recommendations - check what data exists
"""

import psycopg2

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

print("🔍 Debugging Recommendations Data...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Check total interactions
    cur.execute('SELECT COUNT(*) FROM "UserInteraction"')
    total_interactions = cur.fetchone()[0]
    print(f"\n📊 Total Interactions: {total_interactions:,}")
    
    # Check interactions by context
    cur.execute('''
        SELECT "contextId", COUNT(*) as count
        FROM "UserInteraction"
        GROUP BY "contextId"
        ORDER BY "contextId"
    ''')
    print("\n📊 Interactions by Context:")
    for context_id, count in cur.fetchall():
        print(f"  Context {context_id}: {count:,} interactions")
    
    # Check most popular products overall
    cur.execute('''
        SELECT p.id, p.name, COUNT(*) as interaction_count
        FROM "UserInteraction" ui
        JOIN "Product" p ON ui."productId" = p.id
        GROUP BY p.id, p.name
        ORDER BY interaction_count DESC
        LIMIT 10
    ''')
    print("\n🔥 Top 10 Most Popular Products (Overall):")
    for product_id, name, count in cur.fetchall():
        print(f"  {name[:50]}: {count} interactions")
    
    # Check products by specific contexts
    for context_id in [0, 1, 2, 3, 4, 5]:
        cur.execute('''
            SELECT p.id, p.name, COUNT(*) as interaction_count
            FROM "UserInteraction" ui
            JOIN "Product" p ON ui."productId" = p.id
            WHERE ui."contextId" = %s
            GROUP BY p.id, p.name
            ORDER BY interaction_count DESC
            LIMIT 5
        ''', (context_id,))
        
        results = cur.fetchall()
        if results:
            print(f"\n🎯 Top 5 Products for Context {context_id}:")
            for product_id, name, count in results:
                print(f"  {name[:50]}: {count} interactions")
    
    # Check test user
    cur.execute('''
        SELECT id, email, segment, "interactionCount"
        FROM "User"
        WHERE email = 'test@cofars.com'
    ''')
    test_user = cur.fetchone()
    if test_user:
        print(f"\n👤 Test User:")
        print(f"  Email: {test_user[1]}")
        print(f"  Segment: {test_user[2]}")
        print(f"  Interaction Count: {test_user[3]}")
    
    # Check admin user
    cur.execute('''
        SELECT id, email, segment, "interactionCount"
        FROM "User"
        WHERE email = 'admin@cofars.com'
    ''')
    admin_user = cur.fetchone()
    if admin_user:
        print(f"\n👤 Admin User:")
        print(f"  Email: {admin_user[1]}")
        print(f"  Segment: {admin_user[2]}")
        print(f"  Interaction Count: {admin_user[3]}")
        
        # Check admin's interactions
        cur.execute('''
            SELECT COUNT(*), "contextId"
            FROM "UserInteraction"
            WHERE "userId" = %s
            GROUP BY "contextId"
            ORDER BY "contextId"
        ''', (admin_user[0],))
        admin_interactions = cur.fetchall()
        if admin_interactions:
            print(f"  Interactions by context:")
            for count, context_id in admin_interactions:
                print(f"    Context {context_id}: {count} interactions")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    cur.close()
    conn.close()
