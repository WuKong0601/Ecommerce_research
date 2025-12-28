"""
Update product images with inline SVG data URIs
No external dependencies, works offline
"""

import psycopg2
import urllib.parse

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

# Category to color mapping
CATEGORY_COLORS = {
    'Dụng cụ & Thiết bị tiện ích': '#4A90E2',
    'Bảo vệ nhà cửa khác': '#E74C3C',
    'Nhà Cửa - Đời Sống': '#2ECC71',
    'Bàn ghế làm việc': '#9B59B6',
    'Dụng cụ ngoài trời khác': '#F39C12',
    'Thiết bị điện gia dụng': '#3498DB',
    'Đồ dùng nhà bếp': '#E67E22',
    'Nội thất': '#1ABC9C',
    'Trang trí nhà cửa': '#E91E63',
    'Đèn': '#FFC107',
    'Vệ sinh nhà cửa': '#00BCD4',
    'Chăm sóc nhà cửa': '#8BC34A',
}

def create_svg_data_uri(color, category_name):
    """Create an inline SVG with gradient and category icon"""
    # Create a nice gradient SVG
    svg = f'''<svg width="800" height="800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{color};stop-opacity:0.7" />
    </linearGradient>
  </defs>
  <rect width="800" height="800" fill="url(#grad)"/>
  <text x="400" y="420" font-family="Arial, sans-serif" font-size="48" fill="white" text-anchor="middle" opacity="0.9">{category_name[:20]}</text>
</svg>'''
    
    # Encode to data URI
    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml,{encoded}"

def get_color_for_category(category):
    """Get color for category"""
    return CATEGORY_COLORS.get(category, '#95A5A6')

print("🖼️  Updating product images with inline SVG...")

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
        
        # Create inline SVG data URI - works offline!
        image_url = create_svg_data_uri(color, category)
        
        cur.execute(
            'UPDATE "Product" SET "imageUrl" = %s WHERE id = %s',
            (image_url, product_id)
        )
        
        updated_count += 1
        if updated_count % 1000 == 0:
            conn.commit()
            print(f"  Updated {updated_count}/{len(products)} products...")
    
    conn.commit()
    print(f"✅ Updated {updated_count} product images with inline SVG")
    
    # Show sample
    cur.execute('SELECT name, category, LENGTH("imageUrl") FROM "Product" LIMIT 3')
    print("\n📸 Sample products:")
    for name, cat, url_len in cur.fetchall():
        print(f"  {name[:40]} ({cat})")
        print(f"    SVG data URI length: {url_len} bytes")

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
