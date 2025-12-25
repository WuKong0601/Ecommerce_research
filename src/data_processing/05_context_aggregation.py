"""
Step 5: Context Aggregation for CoFARS-Sparse
Build context-level preference profiles by aggregating all users

This script:
1. Loads reviews and user segments  
2. Aggregates user interactions per context
3. Builds context-level preference profiles
4. Creates context embeddings
5. Saves for model training
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.logger import setup_logger

def load_data(config):
    """Load reviews, products, and user segments"""
    logger.info("Loading data...")
    
    contexts_dir = os.path.join(config['data']['processed_dir'], 'contexts')
    segmentation_dir = os.path.join(config['data']['processed_dir'], 'segmentation')
    
    reviews = pd.read_csv(os.path.join(contexts_dir, 'reviews_with_contexts.csv'))
    products = pd.read_csv(os.path.join(contexts_dir, 'products_with_attributes.csv'))
    user_segments = pd.read_csv(os.path.join(segmentation_dir, 'user_segments.csv'))
    
    logger.info(f"Loaded {len(reviews):,} reviews, {len(products):,} products, {len(user_segments):,} users")
    
    return reviews, products, user_segments

def aggregate_context_profiles(reviews, products, config):
    """
    Build context-level preference profiles by aggregating ALL users
    
    Returns:
        context_profiles: Dict with context statistics and preferences
    """
    logger.info("Building context-level preference profiles...")
    
    # Merge reviews with product info
    merged = reviews.merge(
        products[['id', 'category', 'price', 'price_bucket', 'rating_level']],
        left_on='product_id',
        right_on='id',
        how='left'
    )
    
    context_profiles = {}
    contexts = sorted(reviews['context'].unique())
    
    for context in contexts:
        context_data = merged[merged['context'] == context]
        
        # Basic statistics
        profile = {
            'context': context,
            'total_users': context_data['customer_id'].nunique(),
            'total_interactions': len(context_data),
            'total_products': context_data['product_id'].nunique(),
            
            # Average metrics
            'avg_rating': float(context_data['rating'].mean()),
            'avg_price': float(context_data['price'].mean()),
            
            # Popular items  
            'top_products': context_data['product_id'].value_counts().head(20).to_dict(),
            
            # Category preferences
            'category_counts': context_data['category'].value_counts().to_dict(),
            'category_dist': (context_data['category'].value_counts() / len(context_data)).to_dict(),
            
            # Price preferences
            'price_bucket_counts': context_data['price_bucket'].value_counts().to_dict(),
            'price_bucket_dist': (context_data['price_bucket'].value_counts() / len(context_data)).to_dict(),
            
            # Rating preferences
            'rating_level_counts': context_data['rating_level'].value_counts().to_dict(),
            'rating_level_dist': (context_data['rating_level'].value_counts() / len(context_data)).to_dict(),
            
            # User segments in this context
            'segment_counts': {}
        }
        
        # Count which user segments interact in this context
        context_users = set(context_data['customer_id'])
        
        context_profiles[context] = profile
    
    logger.info(f"Built profiles for {len(context_profiles)} contexts")
    
    return context_profiles

def add_segment_distribution(context_profiles, reviews, user_segments):
    """Add user segment distribution to context profiles"""
    logger.info("Adding segment distribution to context profiles...")
    
    # Create user_id to segment mapping
    user_to_segment = dict(zip(user_segments['customer_id'], user_segments['segment']))
    
    for context in context_profiles.keys():
        context_data = reviews[reviews['context'] == context]
        
        # Map users to segments
        context_data['segment'] = context_data['customer_id'].map(user_to_segment)
        
        # Count segments
        segment_counts = context_data['segment'].value_counts().to_dict()
        context_profiles[context]['segment_counts'] = segment_counts
        
        # Distribution
        total = sum(segment_counts.values())
        context_profiles[context]['segment_dist'] = {
            k: v/total for k, v in segment_counts.items()
        }
    
    return context_profiles

def create_context_embeddings(context_profiles, config):
    """
    Create initial context embeddings based on aggregated preferences
    These will be refined during model training
    """
    logger.info("Creating initial context embeddings...")
    
    contexts = sorted(context_profiles.keys())
    
    # Build vocabulary
    all_categories = set()
    all_price_buckets = set()
    all_rating_levels = set()
    
    for profile in context_profiles.values():
        all_categories.update(profile['category_dist'].keys())
        all_price_buckets.update(profile['price_bucket_dist'].keys())
        all_rating_levels.update(profile['rating_level_dist'].keys())
    
    all_categories = sorted(all_categories)
    all_price_buckets = sorted(all_price_buckets)
    all_rating_levels = sorted(all_rating_levels)
    
    logger.info(f"Vocabulary: {len(all_categories)} categories, {len(all_price_buckets)} price buckets, {len(all_rating_levels)} rating levels")
    
    # Create embeddings (one-hot style for now)
    context_embeddings = {}
    
    for context in contexts:
        profile = context_profiles[context]
        
        # Build feature vector
        features = []
        
        # Category features
        for cat in all_categories:
            features.append(profile['category_dist'].get(cat, 0.0))
        
        # Price features
        for price in all_price_buckets:
            features.append(profile['price_bucket_dist'].get(price, 0.0))
        
        # Rating features
        for rating in all_rating_levels:
            features.append(profile['rating_level_dist'].get(rating, 0.0))
        
        # Statistical features
        features.extend([
            profile['avg_rating'] / 5.0,  # Normalize
            np.log(profile['total_interactions'] + 1) / 10.0,  # Log-scale
            profile['total_products'] / 10000.0  # Scale
        ])
        
        context_embeddings[context] = np.array(features, dtype=np.float32)
    
    logger.info(f"Created embeddings with {len(features)} dimensions")
    
    return context_embeddings, {
        'categories': all_categories,
        'price_buckets': all_price_buckets,
        'rating_levels': all_rating_levels
    }

def analyze_context_differences(context_profiles):
    """Analyze how contexts differ in their aggregated preferences"""
    logger.info("Analyzing context differences...")
    
    # Compare top categories across contexts
    logger.info("\nTop category per context:")
    for context, profile in sorted(context_profiles.items()):
        top_cat = max(profile['category_dist'].items(), key=lambda x: x[1])
        logger.info(f"  {context:25s}: {top_cat[0][:30]:30s} ({top_cat[1]*100:.1f}%)")
    
    # Price preferences
    logger.info("\nAverage price per context:")
    for context, profile in sorted(context_profiles.items()):
        logger.info(f"  {context:25s}: {profile['avg_price']:,.0f} VND")
    
    # User coverage
    logger.info("\nUser coverage per context:")
    total_users = max(p['total_users'] for p in context_profiles.values())
    for context, profile in sorted(context_profiles.items()):
        coverage = profile['total_users'] / total_users * 100
        logger.info(f"  {context:25s}: {profile['total_users']:,} users ({coverage:.1f}% of max)")

def save_context_aggregation(context_profiles, context_embeddings, vocabulary, config):
    """Save all context aggregation results"""
    logger.info("Saving context aggregation results...")
    
    # Create output directory
    output_dir = os.path.join(config['data']['processed_dir'], 'context_aggregation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save context profiles (JSON)
    # Convert numpy/pandas types to JSON-serializable
    profiles_serializable = {}
    for context, profile in context_profiles.items():
        profiles_serializable[context] = {
            k: (dict(v) if isinstance(v, (Counter, dict)) else 
                float(v) if isinstance(v, (np.float32, np.float64)) else 
                int(v) if isinstance(v, (np.int32, np.int64)) else v)
            for k, v in profile.items()
        }
    
    with open(os.path.join(output_dir, 'context_profiles.json'), 'w') as f:
        json.dump(profiles_serializable, f, indent=2)
    
    # Save context embeddings (numpy)
    contexts = sorted(context_embeddings.keys())
    embeddings_matrix = np.array([context_embeddings[c] for c in contexts])
    np.save(os.path.join(output_dir, 'context_embeddings.npy'), embeddings_matrix)
    
    # Save context order and vocabulary
    with open(os.path.join(output_dir, 'context_order.json'), 'w') as f:
        json.dump(contexts, f, indent=2)
    
    with open(os.path.join(output_dir, 'vocabulary.json'), 'w') as f:
        json.dump(vocabulary, f, indent=2)
    
    # Save statistics
    stats_dir = config['results']['statistics_dir']
    stats_file = os.path.join(stats_dir, 'context_aggregation_stats.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONTEXT AGGREGATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("AGGREGATION STRATEGY:\n")
        f.write("-"*80 + "\n")
        f.write("  Method: Aggregate ALL user interactions per context\n")
        f.write(f"  Contexts: {len(contexts)}\n")
        f.write(f"  Embedding dimension: {embeddings_matrix.shape[1]}\n\n")
        
        for context in contexts:
            profile = context_profiles[context]
            f.write(f"\n{context.upper()}:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Users: {profile['total_users']:,}\n")
            f.write(f"  Interactions: {profile['total_interactions']:,}\n")
            f.write(f"  Products: {profile['total_products']:,}\n")
            f.write(f"  Avg rating: {profile['avg_rating']:.2f}\n")
            f.write(f"  Avg price: {profile['avg_price']:,.0f}\n")
            
            # Top categories
            top_cats = sorted(profile['category_dist'].items(), key=lambda x: x[1], reverse=True)[:3]
            f.write(f"  Top categories:\n")
            for cat, pct in top_cats:
                f.write(f"    - {cat}: {pct*100:.1f}%\n")
    
    logger.info(f"Saved context aggregation to: {output_dir}")
    logger.info(f"Saved statistics to: {stats_file}")

def main():
    """Main context aggregation pipeline"""
    config = load_config()
    global logger
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("STEP 5: CONTEXT AGGREGATION (CoFARS-Sparse)")
    logger.info("="*80)
    
    # Load data
    reviews, products, user_segments = load_data(config)
    
    # Build context profiles
    context_profiles = aggregate_context_profiles(reviews, products, config)
    
    # Add segment distribution
    context_profiles = add_segment_distribution(context_profiles, reviews, user_segments)
    
    # Create context embeddings
    context_embeddings, vocabulary = create_context_embeddings(context_profiles, config)
    
    # Analyze differences
    analyze_context_differences(context_profiles)
    
    # Save results
    save_context_aggregation(context_profiles, context_embeddings, vocabulary, config)
    
    logger.info("="*80)
    logger.info("CONTEXT AGGREGATION COMPLETE!")
    logger.info("="*80)
    
    return context_profiles, context_embeddings

if __name__ == "__main__":
    main()
