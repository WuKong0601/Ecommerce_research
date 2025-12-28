"""
Fix product images - use Lorem Picsum instead of Unsplash
"""

import psycopg2
import hashlib

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

print("🖼️  Fixing product images with Lorem Picsum...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Get all products
    cur.execute('SELECT id, name FROM "Product"')
    products = cur.fetchall()
    
    print(f"Found {len(products)} products")
    
    updated_count = 0
    for product_id, name in products:
        # Create unique seed based on product_id
        seed = abs(hash(product_id)) % 10000
        
        # Use Lorem Picsum - more reliable than Unsplash
        image_url = f"https://picsum.photos/seed/{seed}/800/800"
        
        cur.execute(
            'UPDATE "Product" SET "imageUrl" = %s WHERE id = %s',
            (image_url, product_id)
        )
        
        updated_count += 1
        if updated_count % 1000 == 0:
            conn.commit()
            print(f"  Updated {updated_count}/{len(products)} products...")
    
    conn.commit()
    print(f"✅ Updated {updated_count} product images with Lorem Picsum")
    
    # Verify
    cur.execute('SELECT "imageUrl" FROM "Product" LIMIT 3')
    print("\n📸 Sample URLs:")
    for (url,) in cur.fetchall():
        print(f"  {url}")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
