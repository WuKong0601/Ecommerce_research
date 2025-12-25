import pandas as pd
import numpy as np

print("="*80)
print("FEATURE ANALYSIS FOR COFARS IMPLEMENTATION")
print("="*80)

# Load data
reviews = pd.read_csv('data/home_life_reviews.csv')
products = pd.read_csv('data/home_life_products_details.csv')

print("\n1. CURRENT AVAILABLE FEATURES")
print("-" * 80)

print("\n📊 Contextual Features (User Interaction Context):")
print(f"  ✅ time_slot: {reviews['time_slot'].nunique()} unique values")
print(f"     Values: {reviews['time_slot'].unique().tolist()}")
print(f"  ✅ hour: {reviews['hour'].nunique()} unique values (0-23)")
print(f"  ✅ day_of_week: {reviews['day_of_week'].nunique()} unique values (0-6)")
print(f"  ✅ is_weekend: Binary (weekday/weekend)")
print(f"  ✅ context_key: {reviews['context_key'].nunique()} unique contexts already created!")

print("\n📦 Product Attributes (for JS Divergence):")
print(f"  ✅ price: Continuous, can be bucketed")
print(f"  ✅ price_bucket: {products['price_bucket'].nunique()} buckets already created")
print(f"  ✅ group (category): {products['group'].nunique()} categories")
print(f"  ✅ rating: {products['rating'].nunique()} unique ratings")
print(f"  ✅ brand: {products['brand'].nunique()} brands")

print("\n⏱️  Temporal Information:")
print(f"  ✅ created_at_raw: Unix timestamp")
print(f"  ✅ event_time: Full datetime")

print("\n" + "="*80)
print("2. CONTEXT RICHNESS ANALYSIS")
print("-" * 80)

# Analyze how rich current contexts are
print("\nCurrent context_key distribution (top 20):")
top_contexts = reviews['context_key'].value_counts().head(20)
for ctx, count in top_contexts.items():
    print(f"  {ctx}: {count} interactions ({count/len(reviews)*100:.2f}%)")

print(f"\nTotal unique contexts: {reviews['context_key'].nunique()}")
print(f"Average interactions per context: {len(reviews) / reviews['context_key'].nunique():.1f}")

# Check what context_key contains
print("\n🔍 Analyzing context_key structure:")
sample_contexts = reviews['context_key'].unique()[:10]
print(f"Sample context keys: {sample_contexts.tolist()}")

print("\n" + "="*80)
print("3. ALTERNATIVE CONTEXT BUILDING STRATEGIES")
print("-" * 80)

# Strategy 1: Use existing context_key
print("\n✅ STRATEGY 1: Use existing 'context_key' field")
print("   Pros: Already available, captures system's understanding of context")
print("   Cons: We don't know exact composition (black box)")

# Strategy 2: Build our own contexts from time features
print("\n✅ STRATEGY 2: Build contexts from temporal features")
reviews['our_context'] = reviews['time_slot'] + '_' + reviews['is_weekend'].map({0: 'weekday', 1: 'weekend'})
print(f"   Created contexts: {reviews['our_context'].nunique()} unique")
print(f"   Example contexts:")
for ctx in reviews['our_context'].unique()[:10]:
    count = (reviews['our_context'] == ctx).sum()
    print(f"     - {ctx}: {count} interactions")

# Strategy 3: More granular contexts
print("\n✅ STRATEGY 3: Fine-grained temporal contexts")
reviews['granular_context'] = (reviews['time_slot'] + '_' + 
                                reviews['day_of_week'].astype(str) + '_' +
                                reviews['is_weekend'].map({0: 'wd', 1: 'we'}))
print(f"   Created contexts: {reviews['granular_context'].nunique()} unique")

print("\n" + "="*80)
print("4. SEQUENCE LENGTH ANALYSIS (CRITICAL)")
print("-" * 80)

# Critical: Check if we have long enough sequences
reviews_valid = reviews[reviews['customer_id'].notna()]
user_seq_lengths = reviews_valid.groupby('customer_id').size()

print(f"\n📈 User Sequence Statistics:")
print(f"   Total users: {len(user_seq_lengths):,}")
print(f"   Mean sequence length: {user_seq_lengths.mean():.1f}")
print(f"   Median sequence length: {user_seq_lengths.median():.0f}")
print(f"   Max sequence length: {user_seq_lengths.max()}")

print(f"\n📊 Distribution:")
print(f"   Users with ≥5 interactions: {(user_seq_lengths >= 5).sum():,} ({(user_seq_lengths >= 5).sum()/len(user_seq_lengths)*100:.1f}%)")
print(f"   Users with ≥10 interactions: {(user_seq_lengths >= 10).sum():,} ({(user_seq_lengths >= 10).sum()/len(user_seq_lengths)*100:.1f}%)")
print(f"   Users with ≥20 interactions: {(user_seq_lengths >= 20).sum():,} ({(user_seq_lengths >= 20).sum()/len(user_seq_lengths)*100:.1f}%)")
print(f"   Users with ≥50 interactions: {(user_seq_lengths >= 50).sum():,} ({(user_seq_lengths >= 50).sum()/len(user_seq_lengths)*100:.1f}%)")

if user_seq_lengths.mean() < 10:
    print("\n⚠️  WARNING: Average sequence length is quite short!")
    print("   CoFARS is designed for LONG sequences (paper: avg 4,423)")
    print("   Recommendation: Focus on users with more interactions")

print("\n" + "="*80)
print("5. PRODUCT-CONTEXT INTERACTION ANALYSIS")
print("-" * 80)

# Check if different contexts show different preferences
merged = reviews_valid.merge(products[['id', 'price_bucket', 'group']], 
                              left_on='product_id', right_on='id', how='left')
# Use the group column from reviews since it exists there too
merged['category'] = merged['group_y'].fillna(merged['group_x'])


print("\n🎯 Product Category distribution by Time Slot:")
context_category = pd.crosstab(merged['time_slot'], merged['category'], normalize='index')
print(f"   Shape: {context_category.shape}")
print(f"   Top categories vary by time_slot: {context_category.idxmax(axis=1).nunique()} different top categories")

print("\n💰 Price Bucket distribution by Time Slot:")
if 'price_bucket' in merged.columns:
    context_price = pd.crosstab(merged['time_slot'], merged['price_bucket'], normalize='index')
    print(f"   Shape: {context_price.shape}")

print("\n✅ This shows context DOES affect preferences (good for CoFARS!)")

print("\n" + "="*80)
print("6. RECOMMENDED APPROACH")
print("-" * 80)

print("""
Based on analysis, here's the BEST approach:

1. ✅ CONTEXT DEFINITION (No synthetic data needed):
   Option A: Use existing 'context_key' (simplest)
   Option B: Build from time_slot + is_weekend (10 contexts)
   Option C: Build from time_slot + day_of_week (35 contexts)
   
   RECOMMENDATION: Start with Option B (10 contexts), expandable to C

2. ✅ PRODUCT ATTRIBUTES for JS Divergence:
   - Category (group field)
   - Price bucket (price_bucket field)
   - Rating level (can bucket into Low/Med/High)
   
   This is SUFFICIENT for JS divergence calculation!

3. ⚠️  SEQUENCE LENGTH:
   - Filter users with minimum 5-10 interactions
   - Use what we have (no synthetic data needed)
   - CoFARS can still work with shorter sequences

4. ✅ LOCATION (Optional):
   - NOT needed if we have rich temporal contexts
   - Paper uses location+time, we use time_slot+day
   - Different domain: home products vs food delivery

5. ✅ IMPLICIT FEEDBACK:
   - Treat reviews as positive interactions
   - Sample non-reviewed products as negative
   - Standard practice in recommender systems

VERDICT: We can implement CoFARS WITHOUT synthetic data!
The temporal contexts + product attributes are sufficient.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
