"""
Seed database with all trained model data
Import products, users, reviews, interactions from processed datasets
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import json
from datetime import datetime
import os
from pathlib import Path
import bcrypt
import uuid

# Database connection
DB_CONFIG = {
    'dbname': 'cofars_ecommerce',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

# Paths to data files
BASE_DIR = Path(__file__).parent.parent.parent.parent
PRODUCTS_FILE = BASE_DIR / 'processed_data/contexts/products_with_attributes.csv'
REVIEWS_FILE = BASE_DIR / 'processed_data/contexts/reviews_with_contexts.csv'
USER_SEGMENTS_FILE = BASE_DIR / 'processed_data/segmentation/user_segments.csv'
CONTEXT_PROFILES_FILE = BASE_DIR / 'processed_data/context_aggregation/context_profiles.json'

print("🚀 Starting database seeding...")
print(f"Base directory: {BASE_DIR}")

# Connect to database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

try:
    # ==================== 1. IMPORT PRODUCTS ====================
    print("\n📦 Importing products...")
    products_df = pd.read_csv(PRODUCTS_FILE)
    
    # Clean and prepare product data
    products_data = []
    for _, row in products_df.iterrows():
        product_id = str(row['id'])
        name = str(row['name'])[:255] if pd.notna(row['name']) else 'Unknown Product'
        description = str(row['short_description']) if pd.notna(row['short_description']) else ''
        # Limit price to max 99999999.99 (Decimal(10,2) limit)
        raw_price = float(row['price']) if pd.notna(row['price']) else 0.0
        price = min(raw_price, 99999999.99)
        category = str(row['category']) if pd.notna(row['category']) else 'Uncategorized'
        group = str(row['group']) if pd.notna(row['group']) else 'Other'
        
        # Map price_bucket string to int (1-5)
        price_bucket_str = str(row['price_bucket']) if pd.notna(row['price_bucket']) else 'low_<100k'
        price_bucket_map = {
            'low_<100k': 1,
            'medium_100k-500k': 2,
            'medium_500k-1m': 3,
            'high_1m-5m': 4,
            'very_high_>5m': 5
        }
        price_bucket = price_bucket_map.get(price_bucket_str, 1)
        
        # Map rating_level to int (1-3)
        rating_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Unknown': 1}
        rating_level_str = str(row['rating_level']) if pd.notna(row['rating_level']) else 'Unknown'
        rating_level = rating_map.get(rating_level_str, 1)
        
        stock = 100  # Default stock
        image_url = None
        is_active = True
        now = datetime.now()
        
        products_data.append((
            product_id, name, description, price, category, group,
            price_bucket, rating_level, stock, image_url, [], is_active, now, now
        ))
    
    # Insert products
    insert_query = """
        INSERT INTO "Product" (id, name, description, price, category, "group", 
                               "priceBucket", "ratingLevel", stock, "imageUrl", images, "isActive",
                               "createdAt", "updatedAt")
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    execute_values(cur, insert_query, products_data)
    conn.commit()
    print(f"✅ Imported {len(products_data)} products")
    
    # ==================== 2. IMPORT USERS ====================
    print("\n👥 Importing users...")
    user_segments_df = pd.read_csv(USER_SEGMENTS_FILE)
    
    # Create users with hashed password
    default_password = bcrypt.hashpw('password123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    users_data = []
    for _, row in user_segments_df.iterrows():
        customer_id = str(row['customer_id'])
        interaction_count = int(row['interaction_count'])
        segment_raw = str(row['segment']).lower()
        
        # Map segment
        if segment_raw == 'cold_start':
            segment = 'COLD_START'
        elif segment_raw == 'regular':
            segment = 'REGULAR'
        elif segment_raw == 'power':
            segment = 'POWER'
        else:
            segment = 'COLD_START'
        
        email = f"user{customer_id}@cofars.com"
        name = f"User {customer_id}"
        role = 'USER'
        now = datetime.now()
        
        users_data.append((
            customer_id, email, default_password, name, None, role,
            segment, interaction_count, now, now
        ))
    
    # Insert users
    insert_query = """
        INSERT INTO "User" (id, email, password, name, phone, role, segment, "interactionCount", "createdAt", "updatedAt")
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    execute_values(cur, insert_query, users_data)
    conn.commit()
    print(f"✅ Imported {len(users_data)} users")
    
    # ==================== 3. IMPORT REVIEWS & INTERACTIONS ====================
    print("\n⭐ Importing reviews and interactions...")
    reviews_df = pd.read_csv(REVIEWS_FILE)
    
    reviews_data = []
    interactions_data = []
    
    for idx, row in reviews_df.iterrows():
        product_id = str(row['product_id'])
        customer_id = str(row['customer_id'])
        rating = int(row['rating']) if pd.notna(row['rating']) else 3
        comment = str(row['content']) if pd.notna(row['content']) else None
        
        # Parse timestamp
        try:
            timestamp = pd.to_datetime(row['event_time'])
        except:
            timestamp = datetime.now()
        
        # Context info
        time_slot = str(row['time_slot']) if pd.notna(row['time_slot']) else 'unknown'
        is_weekend = bool(row['is_weekend']) if pd.notna(row['is_weekend']) else False
        
        # Map context to context_id (0-9)
        context_map = {
            'morning_weekday': 0,
            'morning_weekend': 1,
            'afternoon_weekday': 2,
            'afternoon_weekend': 3,
            'evening_weekday': 4,
            'evening_weekend': 5,
            'late_night_weekday': 6,
            'late_night_weekend': 7,
            'unknown_weekday': 8,
            'unknown_weekend': 9
        }
        context_key = f"{time_slot}_{'weekend' if is_weekend else 'weekday'}"
        context_id = context_map.get(context_key, 8)
        
        # Review data
        review_id = str(uuid.uuid4())
        reviews_data.append((
            review_id, customer_id, product_id, rating, comment, timestamp, timestamp
        ))
        
        # Interaction data (REVIEW type)
        interaction_id = str(uuid.uuid4())
        interactions_data.append((
            interaction_id, customer_id, product_id, 'REVIEW', context_id, time_slot, is_weekend, timestamp
        ))
    
    # Insert reviews
    insert_query = """
        INSERT INTO "Review" (id, "userId", "productId", rating, comment, "createdAt", "updatedAt")
        VALUES %s
        ON CONFLICT ("userId", "productId") DO NOTHING
    """
    execute_values(cur, insert_query, reviews_data)
    conn.commit()
    print(f"✅ Imported {len(reviews_data)} reviews")
    
    # Insert interactions
    insert_query = """
        INSERT INTO "UserInteraction" (id, "userId", "productId", type, "contextId", "timeSlot", "isWeekend", timestamp)
        VALUES %s
    """
    execute_values(cur, insert_query, interactions_data)
    conn.commit()
    print(f"✅ Imported {len(interactions_data)} interactions")
    
    # ==================== 4. IMPORT CONTEXT PROTOTYPES ====================
    print("\n🧠 Importing context prototypes...")
    
    if CONTEXT_PROFILES_FILE.exists():
        with open(CONTEXT_PROFILES_FILE, 'r') as f:
            context_profiles = json.load(f)
        
        context_data = []
        context_names = [
            'morning_weekday', 'morning_weekend',
            'afternoon_weekday', 'afternoon_weekend',
            'evening_weekday', 'evening_weekend',
            'late_night_weekday', 'late_night_weekend',
            'unknown_weekday', 'unknown_weekend'
        ]
        
        for idx, name in enumerate(context_names):
            parts = name.split('_')
            # Handle late_night case (has 3 parts)
            if len(parts) == 3:
                time_slot = f"{parts[0]}_{parts[1]}"  # late_night
                is_weekend = parts[2] == 'weekend'
            else:
                time_slot = parts[0]
                is_weekend = parts[1] == 'weekend'
            
            # Get embedding if available
            embedding = None
            if name in context_profiles:
                embedding = json.dumps(context_profiles[name])
            
            now = datetime.now()
            context_data.append((
                idx, name, time_slot, is_weekend, embedding, 0, now
            ))
        
        # Insert context prototypes
        insert_query = """
            INSERT INTO "ContextPrototype" (id, name, "timeSlot", "isWeekend", embedding, "interactionCount", "updatedAt")
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                "updatedAt" = CURRENT_TIMESTAMP
        """
        execute_values(cur, insert_query, context_data)
        conn.commit()
        print(f"✅ Imported {len(context_data)} context prototypes")
    
    # ==================== 5. UPDATE STATISTICS ====================
    print("\n📊 Updating statistics...")
    
    # Update interaction counts in context prototypes
    cur.execute("""
        UPDATE "ContextPrototype" cp
        SET "interactionCount" = (
            SELECT COUNT(*)
            FROM "UserInteraction" ui
            WHERE ui."contextId" = cp.id
        )
    """)
    
    # Update user interaction counts
    cur.execute("""
        UPDATE "User" u
        SET "interactionCount" = (
            SELECT COUNT(*)
            FROM "UserInteraction" ui
            WHERE ui."userId" = u.id
        )
    """)
    
    # Update user segments based on interaction count
    cur.execute("""
        UPDATE "User"
        SET segment = CASE
            WHEN "interactionCount" >= 5 THEN 'POWER'::"UserSegment"
            WHEN "interactionCount" >= 2 THEN 'REGULAR'::"UserSegment"
            ELSE 'COLD_START'::"UserSegment"
        END
    """)
    
    conn.commit()
    print("✅ Statistics updated")
    
    # ==================== 6. PRINT SUMMARY ====================
    print("\n" + "="*60)
    print("📈 DATABASE SEEDING SUMMARY")
    print("="*60)
    
    cur.execute('SELECT COUNT(*) FROM "Product"')
    product_count = cur.fetchone()[0]
    print(f"Products: {product_count}")
    
    cur.execute('SELECT COUNT(*) FROM "User"')
    user_count = cur.fetchone()[0]
    print(f"Users: {user_count}")
    
    cur.execute('SELECT COUNT(*) FROM "Review"')
    review_count = cur.fetchone()[0]
    print(f"Reviews: {review_count}")
    
    cur.execute('SELECT COUNT(*) FROM "UserInteraction"')
    interaction_count = cur.fetchone()[0]
    print(f"Interactions: {interaction_count}")
    
    cur.execute('SELECT COUNT(*) FROM "ContextPrototype"')
    context_count = cur.fetchone()[0]
    print(f"Context Prototypes: {context_count}")
    
    print("\n📊 User Segments:")
    cur.execute("""
        SELECT segment, COUNT(*) 
        FROM "User" 
        GROUP BY segment 
        ORDER BY segment
    """)
    for segment, count in cur.fetchall():
        print(f"  {segment}: {count}")
    
    print("\n✅ Database seeding completed successfully!")
    print("="*60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    conn.rollback()
    raise

finally:
    cur.close()
    conn.close()
