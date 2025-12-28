"""
Add placeholder images to products using picsum.photos
"""

import psycopg2
import random

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

print("🖼️  Adding product images...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Get all products
    cur.execute('SELECT id FROM "Product" WHERE "imageUrl" IS NULL')
    products = cur.fetchall()
    
    print(f"Found {len(products)} products without images")
    
    # Update products with placeholder images
    for idx, (product_id,) in enumerate(products):
        # Use picsum.photos for random product images
        # Different seed for each product to get variety
        seed = abs(hash(product_id)) % 1000
        image_url = f"https://picsum.photos/seed/{seed}/800/800"
        
        cur.execute(
            'UPDATE "Product" SET "imageUrl" = %s WHERE id = %s',
            (image_url, product_id)
        )
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(products)} products...")
    
    conn.commit()
    print(f"✅ Added images to {len(products)} products")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
