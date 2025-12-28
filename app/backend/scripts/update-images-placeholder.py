"""
Update product images with reliable placeholder service
Uses placeholder.com with category-based colors
"""

import psycopg2

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

# Category to color mapping
CATEGORY_COLORS = {
    'Dụng cụ & Thiết bị tiện ích': '4A90E2',  # Blue
    'Bảo vệ nhà cửa khác': 'E74C3C',  # Red
    'Nhà Cửa - Đời Sống': '2ECC71',  # Green
    'Bàn ghế làm việc': '9B59B6',  # Purple
    'Dụng cụ ngoài trời khác': 'F39C12',  # Orange
    'Thiết bị điện gia dụng': '3498DB',  # Light Blue
    'Đồ dùng nhà bếp': 'E67E22',  # Dark Orange
    'Nội thất': '1ABC9C',  # Turquoise
    'Trang trí nhà cửa': 'E91E63',  # Pink
    'Đèn': 'FFC107',  # Amber
    'Vệ sinh nhà cửa': '00BCD4',  # Cyan
    'Chăm sóc nhà cửa': '8BC34A',  # Light Green
}

def get_color_for_category(category):
    """Get color hex for category"""
    return CATEGORY_COLORS.get(category, '95A5A6')  # Default gray

print("🖼️  Updating product images with placeholder.com...")

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
        
        # Create short product name for display
        short_name = name[:30] if len(name) > 30 else name
        
        # Use placeholder.com - very reliable
        image_url = f"https://via.placeholder.com/800x800/{color}/FFFFFF?text={short_name}"
        
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
    
    # Show samples
    cur.execute('SELECT name, category, "imageUrl" FROM "Product" LIMIT 5')
    print("\n📸 Sample products:")
    for name, cat, url in cur.fetchall():
        print(f"  {name[:40]} ({cat})")
        print(f"    → {url[:80]}...")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
