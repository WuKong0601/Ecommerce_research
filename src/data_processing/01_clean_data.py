"""
Step 1: Data Cleaning
Clean raw data and save to processed_data/cleaned/

This script:
1. Loads raw data
2. Handles missing values
3. Removes duplicates
4. Validates data types
5. Filters invalid records
6. Saves cleaned data with statistics
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.logger import setup_logger

def load_raw_data(config):
    """Load raw data from CSV files"""
    logger.info("Loading raw data...")
    
    products_path = os.path.join(config['data']['raw_dir'], config['data']['products_file'])
    reviews_path = os.path.join(config['data']['raw_dir'], config['data']['reviews_file'])
    
    products = pd.read_csv(products_path)
    reviews = pd.read_csv(reviews_path)
    
    logger.info(f"Loaded {len(products):,} products and {len(reviews):,} reviews")
    
    return products, reviews

def clean_products(products):
    """Clean products data"""
    logger.info("Cleaning products data...")
    
    initial_count = len(products)
    stats = {'initial_count': initial_count}
    
    # 1. Check for duplicates
    duplicates = products.duplicated(subset=['id'])
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate product IDs - keeping first occurrence")
        products = products.drop_duplicates(subset=['id'], keep='first')
        stats['duplicates_removed'] = duplicates.sum()
    
    # 2. Handle missing values
    logger.info("Handling missing values in products...")
    
    # Essential columns that should not be missing
    essential_cols = ['id', 'name', 'price']
    missing_essential = products[essential_cols].isnull().any(axis=1)
    
    if missing_essential.any():
        logger.warning(f"Removing {missing_essential.sum()} products with missing essential data")
        products = products[~missing_essential]
        stats['missing_essential'] = missing_essential.sum()
    
    # Fill missing values for optional columns
    products['rating'] = products['rating'].fillna(0)  # No rating yet
    products['review_count'] = products['review_count'].fillna(0)
    products['brand'] = products['brand'].fillna('Unknown')
    products['price_bucket'] = products['price_bucket'].fillna('Unknown')
    
    # 3. Validate price (must be positive)
    invalid_price = products['price'] <= 0
    if invalid_price.any():
        logger.warning(f"Removing {invalid_price.sum()} products with invalid price")
        products = products[~invalid_price]
        stats['invalid_price'] = invalid_price.sum()
    
    # 4. Validate data types
    products['id'] = products['id'].astype(int)
    products['price'] = products['price'].astype(float)
    products['review_count'] = products['review_count'].astype(int)
    
    stats['final_count'] = len(products)
    stats['removed_total'] = initial_count - len(products)
    
    logger.info(f"Products cleaning complete: {initial_count:,} → {len(products):,} (-{stats['removed_total']:,})")
    
    return products, stats

def clean_reviews(reviews):
    """Clean reviews data"""
    logger.info("Cleaning reviews data...")
    
    initial_count = len(reviews)
    stats = {'initial_count': initial_count}
    
    # 1. Check for duplicates (same user, product, time)
    duplicates = reviews.duplicated(subset=['product_id', 'customer_id', 'created_at_raw'])
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate reviews - keeping first occurrence")
        reviews = reviews.drop_duplicates(subset=['product_id', 'customer_id', 'created_at_raw'], keep='first')
        stats['duplicates_removed'] = duplicates.sum()
    
    # 2. Handle missing values
    logger.info("Handling missing values in reviews...")
    
    # Essential columns
    essential_cols = ['product_id', 'rating', 'created_at_raw']
    missing_essential = reviews[essential_cols].isnull().any(axis=1)
    
    if missing_essential.any():
        logger.warning(f"Removing {missing_essential.sum()} reviews with missing essential data")
        reviews = reviews[~missing_essential]
        stats['missing_essential'] = missing_essential.sum()
    
    # Handle missing customer_id (anonymous reviews)
    missing_customer = reviews['customer_id'].isnull()
    if missing_customer.any():
        logger.warning(f"Removing {missing_customer.sum()} reviews with missing customer_id (cannot build sequences)")
        reviews = reviews[~missing_customer]
        stats['missing_customer_id'] = missing_customer.sum()
    
    # Fill missing context features
    reviews['time_slot'] = reviews['time_slot'].fillna('unknown')
    reviews['is_weekend'] = reviews['is_weekend'].fillna(0)
    reviews['day_of_week'] = reviews['day_of_week'].fillna(-1)
    
    # 3. Validate rating (1-5 scale)
    invalid_rating = (reviews['rating'] < 1) | (reviews['rating'] > 5)
    if invalid_rating.any():
        logger.warning(f"Removing {invalid_rating.sum()} reviews with invalid rating")
        reviews = reviews[~invalid_rating]
        stats['invalid_rating'] = invalid_rating.sum()
    
    # 4. Validate timestamps
    invalid_timestamp = reviews['created_at_raw'] <= 0
    if invalid_timestamp.any():
        logger.warning(f"Removing {invalid_timestamp.sum()} reviews with invalid timestamp")
        reviews = reviews[~invalid_timestamp]
        stats['invalid_timestamp'] = invalid_timestamp.sum()
    
    # 5. Validate time_slot values
    valid_time_slots = ['morning', 'afternoon', 'evening', 'night', 'late_night', 'unknown']
    invalid_time_slot = ~reviews['time_slot'].isin(valid_time_slots)
    if invalid_time_slot.any():
        logger.warning(f"Found {invalid_time_slot.sum()} reviews with invalid time_slot - setting to 'unknown'")
        reviews.loc[invalid_time_slot, 'time_slot'] = 'unknown'
        stats['invalid_time_slot'] = invalid_time_slot.sum()
    
    # 6. Validate data types
    reviews['product_id'] = reviews['product_id'].astype(int)
    reviews['customer_id'] = reviews['customer_id'].astype(int)
    reviews['rating'] = reviews['rating'].astype(int)
    reviews['created_at_raw'] = reviews['created_at_raw'].astype(int)
    reviews['is_weekend'] = reviews['is_weekend'].astype(int)
    
    stats['final_count'] = len(reviews)
    stats['removed_total'] = initial_count - len(reviews)
    
    logger.info(f"Reviews cleaning complete: {initial_count:,} → {len(reviews):,} (-{stats['removed_total']:,})")
    
    return reviews, stats

def filter_valid_interactions(products, reviews):
    """Filter reviews to only include products that exist in products table"""
    logger.info("Filtering valid product-review interactions...")
    
    initial_count = len(reviews)
    valid_product_ids = set(products['id'])
    
    # Keep only reviews for products that exist
    reviews = reviews[reviews['product_id'].isin(valid_product_ids)]
    
    removed = initial_count - len(reviews)
    if removed > 0:
        logger.warning(f"Removed {removed:,} reviews for non-existent products")
    
    logger.info(f"Valid interactions: {len(reviews):,}")
    
    return reviews, {'removed_invalid_products': removed}

def save_cleaned_data(products, reviews, products_stats, reviews_stats, interaction_stats, config):
    """Save cleaned data and statistics"""
    logger.info("Saving cleaned data...")
    
    # Create output directory
    output_dir = os.path.join(config['data']['processed_dir'], 'cleaned')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned data
    products_path = os.path.join(output_dir, 'products_cleaned.csv')
    reviews_path = os.path.join(output_dir, 'reviews_cleaned.csv')
    
    products.to_csv(products_path, index=False)
    reviews.to_csv(reviews_path, index=False)
    
    logger.info(f"Saved cleaned products to: {products_path}")
    logger.info(f"Saved cleaned reviews to: {reviews_path}")
    
    # Save cleaning statistics
    stats_dir = config['results']['statistics_dir']
    os.makedirs(stats_dir, exist_ok=True)
    
    stats_path = os.path.join(stats_dir, 'data_cleaning_stats.txt')
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DATA CLEANING STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Cleaning Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PRODUCTS CLEANING:\n")
        f.write("-"*80 + "\n")
        for key, value in products_stats.items():
            f.write(f"  {key}: {value:,}\n")
        
        f.write("\nREVIEWS CLEANING:\n")
        f.write("-"*80 + "\n")
        for key, value in reviews_stats.items():
            f.write(f"  {key}: {value:,}\n")
        
        f.write("\nINTERACTION FILTERING:\n")
        f.write("-"*80 + "\n")
        for key, value in interaction_stats.items():
            f.write(f"  {key}: {value:,}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL DATA SUMMARY:\n")
        f.write("="*80 + "\n")
        f.write(f"  Total Products: {len(products):,}\n")
        f.write(f"  Total Reviews: {len(reviews):,}\n")
        f.write(f"  Total Users: {reviews['customer_id'].nunique():,}\n")
        f.write(f"  Avg Reviews per User: {len(reviews) / reviews['customer_id'].nunique():.2f}\n")
        f.write(f"  Avg Reviews per Product: {len(reviews) / len(products):.2f}\n")
    
    logger.info(f"Saved cleaning statistics to: {stats_path}")
    
    return products_path, reviews_path, stats_path

def main():
    """Main data cleaning pipeline"""
    # Setup
    config = load_config()
    global logger
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("STEP 1: DATA CLEANING")
    logger.info("="*80)
    
    # Load raw data
    products, reviews = load_raw_data(config)
    
    # Clean products
    products, products_stats = clean_products(products)
    
    # Clean reviews
    reviews, reviews_stats = clean_reviews(reviews)
    
    # Filter valid interactions
    reviews, interaction_stats = filter_valid_interactions(products, reviews)
    
    # Save cleaned data
    products_path, reviews_path, stats_path = save_cleaned_data(
        products, reviews, products_stats, reviews_stats, interaction_stats, config
    )
    
    logger.info("="*80)
    logger.info("DATA CLEANING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Cleaned products: {products_path}")
    logger.info(f"Cleaned reviews: {reviews_path}")
    logger.info(f"Statistics: {stats_path}")
    
    return products, reviews

if __name__ == "__main__":
    main()
