"""
Update product images intelligently based on product names
Uses Unsplash API with smart keyword matching
"""

import psycopg2
import time

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

# Pexels API (free, no key needed for basic usage)
# We'll use Unsplash API instead which is more reliable
def get_image_for_product(product_name, product_id):
    """
    Generate appropriate image URL based on product name
    Using Unsplash API with Vietnamese product keywords
    """
    
    # Extract key product type from Vietnamese product name
    keywords_map = {
        # Furniture
        'tủ': 'cabinet furniture',
        'bàn': 'desk table',
        'ghế': 'chair furniture',
        'giường': 'bed bedroom',
        'kệ': 'shelf storage',
        'sofa': 'sofa couch',
        
        # Kitchen
        'nồi': 'pot cookware',
        'chảo': 'pan cookware',
        'bát': 'bowl dish',
        'đĩa': 'plate dish',
        'ly': 'glass cup',
        'cốc': 'cup mug',
        'dao': 'knife kitchen',
        'thìa': 'spoon cutlery',
        'muỗng': 'spoon cutlery',
        
        # Home appliances
        'quạt': 'fan electric',
        'đèn': 'lamp light',
        'máy': 'machine appliance',
        'bình': 'bottle container',
        
        # Cleaning
        'chổi': 'broom cleaning',
        'lau': 'mop cleaning',
        'giẻ': 'cloth cleaning',
        'xô': 'bucket cleaning',
        
        # Storage
        'hộp': 'box container',
        'túi': 'bag storage',
        'giỏ': 'basket storage',
        'thùng': 'bin container',
        
        # Decoration
        'tranh': 'painting art',
        'gương': 'mirror decor',
        'rèm': 'curtain window',
        'thảm': 'rug carpet',
        'gối': 'pillow cushion',
        'chăn': 'blanket bedding',
        
        # Tools
        'kéo': 'scissors tool',
        'búa': 'hammer tool',
        'vít': 'screw tool',
        'móc': 'hook hanger',
        
        # Others
        'nhang': 'incense',
        'nến': 'candle',
        'chuông': 'bell doorbell',
    }
    
    # Find matching keyword
    search_term = 'home-product'  # default
    product_lower = product_name.lower()
    
    for vn_word, en_term in keywords_map.items():
        if vn_word in product_lower:
            search_term = en_term.replace(' ', '-')
            break
    
    # Use Unsplash with specific search term and product ID as seed
    seed = abs(hash(product_id)) % 10000
    image_url = f"https://source.unsplash.com/800x800/?{search_term}&sig={seed}"
    
    return image_url

print("🖼️  Updating product images intelligently...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Get all products
    cur.execute('SELECT id, name FROM "Product" ORDER BY id')
    products = cur.fetchall()
    
    print(f"Found {len(products)} products")
    
    updated_count = 0
    for product_id, name in products:
        image_url = get_image_for_product(name, product_id)
        
        cur.execute(
            'UPDATE "Product" SET "imageUrl" = %s WHERE id = %s',
            (image_url, product_id)
        )
        
        updated_count += 1
        if updated_count % 500 == 0:
            conn.commit()
            print(f"  Updated {updated_count}/{len(products)} products...")
            time.sleep(0.5)  # Rate limiting
    
    conn.commit()
    print(f"✅ Updated {updated_count} product images")
    
    # Show samples
    cur.execute('SELECT name, "imageUrl" FROM "Product" LIMIT 10')
    print("\n📸 Sample products:")
    for name, url in cur.fetchall():
        print(f"  {name[:50]}")
        print(f"    → {url}")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
