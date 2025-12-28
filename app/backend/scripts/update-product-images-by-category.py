"""
Update product images based on category using Unsplash API
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

# Map Vietnamese categories to English search terms for better images
CATEGORY_IMAGE_MAP = {
    'Dụng cụ & Thiết bị tiện ích': 'home-tools-kitchen-utensils',
    'Bảo vệ nhà cửa khác': 'home-security-door-lock',
    'Nhà Cửa - Đời Sống': 'home-living-furniture',
    'Bàn ghế làm việc': 'office-desk-chair',
    'Dụng cụ ngoài trời khác': 'outdoor-camping-tools',
    'Thiết bị điện gia dụng': 'home-appliances-electronics',
    'Đồ dùng nhà bếp': 'kitchen-cookware-utensils',
    'Nội thất': 'furniture-interior-design',
    'Trang trí nhà cửa': 'home-decoration-decor',
    'Đèn': 'lighting-lamp-bulb',
    'Vệ sinh nhà cửa': 'cleaning-supplies-home',
    'Chăm sóc nhà cửa': 'home-care-maintenance',
}

print("🖼️  Updating product images by category...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Get all products with their categories
    cur.execute('SELECT id, category, name FROM "Product"')
    products = cur.fetchall()
    
    print(f"Found {len(products)} products")
    
    updated_count = 0
    for product_id, category, name in products:
        # Get search term for category
        search_term = CATEGORY_IMAGE_MAP.get(category, 'home-product')
        
        # Create unique seed based on product_id
        seed = abs(hash(product_id)) % 10000
        
        # Use Unsplash Source API with category-specific search
        # Format: https://source.unsplash.com/800x800/?{search_term}&sig={seed}
        image_url = f"https://source.unsplash.com/800x800/?{search_term}&sig={seed}"
        
        cur.execute(
            'UPDATE "Product" SET "imageUrl" = %s WHERE id = %s',
            (image_url, product_id)
        )
        
        updated_count += 1
        if updated_count % 1000 == 0:
            conn.commit()
            print(f"  Updated {updated_count}/{len(products)} products...")
    
    conn.commit()
    print(f"✅ Updated {updated_count} product images with category-specific images")
    
    # Print category distribution
    print("\n📊 Category Distribution:")
    cur.execute('''
        SELECT category, COUNT(*) as count
        FROM "Product"
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10
    ''')
    for cat, count in cur.fetchall():
        print(f"  {cat}: {count} products")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
