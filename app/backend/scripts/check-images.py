"""
Check and fix product image URLs
"""

import psycopg2

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

print("🔍 Checking product images...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Check sample images
    cur.execute('SELECT id, name, "imageUrl" FROM "Product" LIMIT 5')
    print("\n📸 Sample product images:")
    for product_id, name, image_url in cur.fetchall():
        print(f"  {name[:50]}")
        print(f"    URL: {image_url[:100] if image_url else 'NULL'}")
    
    # Count products with invalid URLs
    cur.execute('''
        SELECT COUNT(*) 
        FROM "Product" 
        WHERE "imageUrl" IS NULL 
           OR "imageUrl" = '' 
           OR "imageUrl" NOT LIKE 'http%'
    ''')
    invalid_count = cur.fetchone()[0]
    print(f"\n❌ Products with invalid/missing images: {invalid_count}")
    
    # Count products with valid URLs
    cur.execute('''
        SELECT COUNT(*) 
        FROM "Product" 
        WHERE "imageUrl" LIKE 'http%'
    ''')
    valid_count = cur.fetchone()[0]
    print(f"✅ Products with valid images: {valid_count}")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    cur.close()
    conn.close()
