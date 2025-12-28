"""
Update product images with solid color backgrounds based on category
No text, just colors - no encoding issues
"""

import psycopg2

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

# Category to color mapping (hex colors)
CATEGORY_COLORS = {
    'Dụng cụ & Thiết bị tiện ích': '4A90E2',
    'Bảo vệ nhà cửa khác': 'E74C3C',
    'Nhà Cửa - Đời Sống': '2ECC71',
    'Bàn ghế làm việc': '9B59B6',
    'Dụng cụ ngoài trời khác': 'F39C12',
    'Thiết bị điện gia dụng': '3498DB',
    'Đồ dùng nhà bếp': 'E67E22',
    'Nội thất': '1ABC9C',
    'Trang trí nhà cửa': 'E91E63',
    'Đèn': 'FFC107',
    'Vệ sinh nhà cửa': '00BCD4',
    'Chăm sóc nhà cửa': '8BC34A',
}

def get_color_for_category(category):
    """Get color hex for category"""
    return CATEGORY_COLORS.get(category, '95A5A6')

print("🖼️  Updating product images with solid colors...")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Get all products with categories
    cur.execute('SELECT id, name, category FROM "Product" ORDER BY id')
    products = cur.fetchall()
    
    print(f"Found {len(products)} products")
    
    updated_count = 0
    for product_id, name, category in products:
        color = get_color_for_category(category)
        
        # Use placeholder.com with just color, no text
        # This avoids encoding issues with Vietnamese text
        image_url = f"https://via.placeholder.com/800x800/{color}/FFFFFF"
        
        cur.execute(
            'UPDATE "Product" SET "imageUrl" = %s WHERE id = %s',
            (image_url, product_id)
        )
        
        updated_count += 1
        if updated_count % 1000 == 0:
            conn.commit()
            print(f"  Updated {updated_count}/{len(products)} products...")
    
    conn.commit()
    print(f"✅ Updated {updated_count} product images")
    
    # Show samples by category
    print("\n📸 Sample by category:")
    for cat, color in CATEGORY_COLORS.items():
        cur.execute(f'SELECT COUNT(*) FROM "Product" WHERE category = %s', (cat,))
        count = cur.fetchone()[0]
        if count > 0:
            print(f"  {cat}: {count} products (Color: #{color})")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
