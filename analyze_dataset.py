import pandas as pd
import numpy as np
import sys

# Redirect output to file
output_file = open('data_analysis_report.txt', 'w', encoding='utf-8')
sys.stdout = output_file

# Load data
print("Loading data...")
reviews = pd.read_csv('data/home_life_reviews.csv')
products = pd.read_csv('data/home_life_products_details.csv')

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total products: {len(products):,}")
print(f"Total reviews: {len(reviews):,}")

print("\n" + "="*80)
print("USER BEHAVIOR SEQUENCE ANALYSIS")
print("="*80)
reviews_valid = reviews[reviews['customer_id'].notna()]
print(f"Total unique users: {reviews_valid['customer_id'].nunique():,}")

user_sequences = reviews_valid.groupby('customer_id').size()
print(f"\nSequence length statistics:")
print(f"  Mean: {user_sequences.mean():.2f}")
print(f"  Median: {user_sequences.median():.0f}")
print(f"  Min: {user_sequences.min()}")
print(f"  Max: {user_sequences.max()}")
print(f"  Std: {user_sequences.std():.2f}")

print(f"\nUser distribution by sequence length:")
print(f"  Users with 1 interaction: {(user_sequences == 1).sum():,}")
print(f"  Users with 2-5 interactions: {((user_sequences >= 2) & (user_sequences <= 5)).sum():,}")
print(f"  Users with 6-10 interactions: {((user_sequences >= 6) & (user_sequences <= 10)).sum():,}")
print(f"  Users with 11-50 interactions: {((user_sequences >= 11) & (user_sequences <= 50)).sum():,}")
print(f"  Users with 51-100 interactions: {((user_sequences >= 51) & (user_sequences <= 100)).sum():,}")
print(f"  Users with >100 interactions: {(user_sequences > 100).sum():,}")

print("\n" + "="*80)
print("CONTEXTUAL FEATURES ANALYSIS")
print("="*80)

print("\n1. Time slot distribution:")
for slot, count in reviews['time_slot'].value_counts().items():
    print(f"  {slot}: {count:,} ({count/len(reviews)*100:.1f}%)")

print("\n2. Day of week distribution:")
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
             4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
for day, count in reviews['day_of_week'].value_counts().sort_index().items():
    day_name = day_names.get(day, f'Day {day}')
    print(f"  {day_name}: {count:,} ({count/len(reviews)*100:.1f}%)")

print("\n3. Weekend vs Weekday:")
for is_we, count in reviews['is_weekend'].value_counts().items():
    label = 'Weekend' if is_we == 1 else 'Weekday'
    print(f"  {label}: {count:,} ({count/len(reviews)*100:.1f}%)")

print("\n4. Total unique context keys:", reviews['context_key'].nunique())
print("Top 10 context keys:")
for ctx, count in reviews['context_key'].value_counts().head(10).items():
    print(f"  {ctx}: {count:,}")

print("\n" + "="*80)
print("PRODUCT ATTRIBUTES ANALYSIS")
print("="*80)

print("\n1. Price statistics:")
print(f"  Mean price: {products['price'].mean():,.0f}")
print(f"  Median price: {products['price'].median():,.0f}")
print(f"  Min price: {products['price'].min():,.0f}")
print(f"  Max price: {products['price'].max():,.0f}")

print("\n2. Price bucket distribution:")
for bucket, count in products['price_bucket'].value_counts().items():
    print(f"  {bucket}: {count:,} ({count/len(products)*100:.1f}%)")

print("\n3. Rating statistics:")
print(f"  Mean rating: {products['rating'].mean():.2f}")
print(f"  Products with rating: {products['rating'].notna().sum():,}")

print("\n4. Product categories (top 10):")
for cat, count in products['group'].value_counts().head(10).items():
    print(f"  {cat}: {count:,} ({count/len(products)*100:.1f}%)")

print("\n5. Total unique categories:", products['group'].nunique())

print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)
print("\nReviews missing values:")
for col in reviews.columns:
    missing = reviews[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing:,} ({missing/len(reviews)*100:.1f}%)")

print("\nProducts missing values:")
for col in products.columns:
    missing = products[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing:,} ({missing/len(products)*100:.1f}%)")

print("\n" + "="*80)
print("TEMPORAL ANALYSIS")
print("="*80)
reviews_sorted = reviews_valid.sort_values('created_at_raw')
print(f"Earliest review: {reviews_sorted.iloc[0]['event_time']}")
print(f"Latest review: {reviews_sorted.iloc[-1]['event_time']}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

output_file.close()
print("Report saved to data_analysis_report.txt", file=sys.__stdout__)
