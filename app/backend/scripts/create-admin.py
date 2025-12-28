"""
Create admin user for testing
"""

import psycopg2
import bcrypt

DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # Create admin user
    admin_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    cur.execute("""
        INSERT INTO "User" (id, email, password, name, phone, role, segment, "interactionCount", "createdAt", "updatedAt")
        VALUES ('admin', 'admin@cofars.com', %s, 'Admin User', NULL, 'ADMIN', 'POWER', 10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (email) DO UPDATE SET
            password = EXCLUDED.password,
            role = 'ADMIN'
    """, (admin_password,))
    
    # Create test user
    test_password = bcrypt.hashpw('test123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    cur.execute("""
        INSERT INTO "User" (id, email, password, name, phone, role, segment, "interactionCount", "createdAt", "updatedAt")
        VALUES ('test_user', 'test@cofars.com', %s, 'Test User', '0123456789', 'USER', 'COLD_START', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (email) DO UPDATE SET
            password = EXCLUDED.password
    """, (test_password,))
    
    conn.commit()
    
    print("✅ Admin and test users created successfully!")
    print("\n📝 Login credentials:")
    print("="*50)
    print("Admin account:")
    print("  Email: admin@cofars.com")
    print("  Password: admin123")
    print("\nTest account:")
    print("  Email: test@cofars.com")
    print("  Password: test123")
    print("="*50)

except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
