"""
Step 2: Build Contexts (Option B: time_slot + is_weekend)
Creates context labels and analyzes context distribution

This script:
1. Loads cleaned data
2. Creates context labels from time_slot + is_weekend
3. Analyzes context distribution
4. Processes product attributes (category, price_bucket, rating)
5. Calculates attribute distributions per context
6. Saves processed data with statistics and visualizations
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.logger import setup_logger

def load_cleaned_data(config):
    """Load cleaned data"""
    logger.info("Loading cleaned data...")
    
    cleaned_dir = os.path.join(config['data']['processed_dir'], 'cleaned')
    products_path = os.path.join(cleaned_dir, 'products_cleaned.csv')
    reviews_path = os.path.join(cleaned_dir, 'reviews_cleaned.csv')
    
    products = pd.read_csv(products_path)
    reviews = pd.read_csv(reviews_path)
    
    logger.info(f"Loaded {len(products):,} products and {len(reviews):,} reviews")
    
    return products, reviews

def create_contexts(reviews, config):
    """
    Create context labels using Option B: time_slot + is_weekend
    
    Returns:
        reviews: DataFrame with 'context' column added
        context_mapping: Dictionary mapping context to its components
    """
    logger.info("Creating contexts (Option B: time_slot + is_weekend)...")
    
    # Create context label
    reviews['context'] = (reviews['time_slot'] + '_' + 
                         reviews['is_weekend'].map({0: 'weekday', 1: 'weekend'}))
    
    # Create context mapping
    context_mapping = {}
    for _, row in reviews[['context', 'time_slot', 'is_weekend']].drop_duplicates().iterrows():
        context_mapping[row['context']] = {
            'time_slot': row['time_slot'],
            'is_weekend': 'weekend' if row['is_weekend'] == 1 else 'weekday'
        }
    
    logger.info(f"Created {len(context_mapping)} unique contexts")
    
    # Print context examples
    logger.info("Context examples:")
    for ctx in sorted(context_mapping.keys())[:5]:
        logger.info(f"  {ctx}: {context_mapping[ctx]}")
    
    return reviews, context_mapping

def analyze_context_distribution(reviews, context_mapping, config):
    """Analyze and visualize context distribution"""
    logger.info("Analyzing context distribution...")
    
    context_counts = reviews['context'].value_counts()
    
    stats = {
        'total_contexts': len(context_mapping),
        'total_interactions': len(reviews),
        'avg_interactions_per_context': len(reviews) / len(context_mapping),
        'min_interactions': context_counts.min(),
        'max_interactions': context_counts.max(),
        'contexts': {}
    }
    
    for context, count in context_counts.items():
        stats['contexts'][context] = {
            'count': int(count),
            'percentage': float(count / len(reviews) * 100)
        }
    
    logger.info(f"Context distribution:")
    logger.info(f"  Total contexts: {stats['total_contexts']}")
    logger.info(f"  Avg interactions per context: {stats['avg_interactions_per_context']:.1f}")
    logger.info(f"  Min/Max: {stats['min_interactions']:,} / {stats['max_interactions']:,}")
    
    # Create visualization
    create_context_distribution_plot(context_counts, config)
    
    return stats

def create_context_distribution_plot(context_counts, config):
    """Create and save context distribution visualization"""
    logger.info("Creating context distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    ax1 = axes[0]
    context_counts.sort_values(ascending=True).plot(kind='barh', ax=ax1)
    ax1.set_xlabel('Number of Interactions')
    ax1.set_ylabel('Context')
    ax1.set_title('Context Distribution')
    ax1.grid(axis='x', alpha=0.3)
    
    # Pie chart
    ax2 = axes[1]
    context_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_ylabel('')
    ax2.set_title('Context Proportion')
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = config['results']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'context_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved context distribution plot to: {save_path}")

def process_product_attributes(products, config):
    """Process and bucket product attributes"""
    logger.info("Processing product attributes...")
    
    # 1. Category (already exists as 'group')
    products['category'] = products['group']
    
    # 2. Price bucket (already exists, but validate)
    if 'price_bucket' not in products.columns or products['price_bucket'].isna().any():
        logger.info("Creating price buckets...")
        # Create price buckets if not exist
        products['price_bucket'] = pd.qcut(
            products['price'], 
            q=config['preprocessing']['price_buckets'],
            labels=[f'price_{i}' for i in range(config['preprocessing']['price_buckets'])],
            duplicates='drop'
        )
    
    # 3. Rating level (bucket into Low/Medium/High)
    logger.info("Creating rating levels...")
    
    # Handle ratings: 0 means no rating, 1-5 are actual ratings
    def bucket_rating(rating):
        if pd.isna(rating) or rating == 0:
            return 'Unknown'
        elif rating <= 2:
            return 'Low'
        elif rating <= 4:
            return 'Medium'
        else:
            return 'High'
    
    products['rating_level'] = products['rating'].apply(bucket_rating)
    
    logger.info("Product attributes processed:")
    logger.info(f"  Categories: {products['category'].nunique()}")
    logger.info(f"  Price buckets: {products['price_bucket'].nunique()}")
    logger.info(f"  Rating levels: {products['rating_level'].nunique()}")
    
    return products

def calculate_context_attribute_distributions(reviews, products, context_mapping, config):
    """
    Calculate attribute distributions for each context
    This is used for ground-truth JS divergence calculation
    """
    logger.info("Calculating attribute distributions per context...")
    
    # Merge reviews with product attributes
    merged = reviews.merge(
        products[['id', 'category', 'price_bucket', 'rating_level']],
        left_on='product_id',
        right_on='id',
        how='left'
    )
    
    context_distributions = {}
    
    for context in sorted(context_mapping.keys()):
        context_data = merged[merged['context'] == context]
        
        if len(context_data) == 0:
            logger.warning(f"Context {context} has no data!")
            continue
        
        # Calculate distributions for each attribute
        distributions = {}
        
        # Category distribution
        category_counts = context_data['category'].value_counts()
        distributions['category'] = (category_counts / category_counts.sum()).to_dict()
        
        # Price bucket distribution
        price_counts = context_data['price_bucket'].value_counts()
        distributions['price_bucket'] = (price_counts / price_counts.sum()).to_dict()
        
        # Rating level distribution
        rating_counts = context_data['rating_level'].value_counts()
        distributions['rating_level'] = (rating_counts / rating_counts.sum()).to_dict()
        
        context_distributions[context] = {
            'count': len(context_data),
            'distributions': distributions
        }
    
    logger.info(f"Calculated distributions for {len(context_distributions)} contexts")
    
    return context_distributions

def create_context_attribute_heatmap(context_distributions, config):
    """Create heatmap visualization of context-attribute relationships"""
    logger.info("Creating context-attribute heatmaps...")
    
    figures_dir = config['results']['figures_dir']
    
    # Get all unique categories
    all_categories = set()
    for ctx_data in context_distributions.values():
        all_categories.update(ctx_data['distributions']['category'].keys())
    all_categories = sorted(all_categories)
    
    # Create matrix for top N categories
    top_n = 10
    category_freq = Counter()
    for ctx_data in context_distributions.values():
        for cat, prob in ctx_data['distributions']['category'].items():
            category_freq[cat] += prob
    top_categories = [cat for cat, _ in category_freq.most_common(top_n)]
    
    # Build heatmap matrix
    matrix = []
    contexts = sorted(context_distributions.keys())
    
    for context in contexts:
        row = []
        for category in top_categories:
            prob = context_distributions[context]['distributions']['category'].get(category, 0)
            row.append(prob)
        matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        matrix,
        xticklabels=[cat[:20] for cat in top_categories],  # Truncate long names
        yticklabels=contexts,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Probability'}
    )
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Context')
    ax.set_title(f'Context-Category Distribution (Top {top_n} Categories)')
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, 'context_category_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved context-category heatmap to: {save_path}")

def save_processed_data(reviews, products, context_mapping, context_stats, 
                        context_distributions, config):
    """Save all processed data"""
    logger.info("Saving processed data...")
    
    # Create output directory
    output_dir = os.path.join(config['data']['processed_dir'], 'contexts')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files
    reviews.to_csv(os.path.join(output_dir, 'reviews_with_contexts.csv'), index=False)
    products.to_csv(os.path.join(output_dir, 'products_with_attributes.csv'), index=False)
    
    # Save context mapping as JSON
    import json
    with open(os.path.join(output_dir, 'context_mapping.json'), 'w') as f:
        json.dump(context_mapping, f, indent=2)
    
    # Save context distributions
    with open(os.path.join(output_dir, 'context_distributions.json'), 'w') as f:
        json.dump(context_distributions, f, indent=2, default=str)
    
    # Save statistics
    stats_dir = config['results']['statistics_dir']
    stats_file = os.path.join(stats_dir, 'context_building_stats.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONTEXT BUILDING STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONTEXT CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Strategy: Option B (time_slot + is_weekend)\n")
        f.write(f"  Total contexts: {context_stats['total_contexts']}\n")
        f.write(f"  Total interactions: {context_stats['total_interactions']:,}\n")
        f.write(f"  Avg interactions/context: {context_stats['avg_interactions_per_context']:.1f}\n\n")
        
        f.write("CONTEXT DISTRIBUTION:\n")
        f.write("-"*80 + "\n")
        for context, data in sorted(context_stats['contexts'].items(), 
                                    key=lambda x: x[1]['count'], reverse=True):
            f.write(f"  {context:25s}: {data['count']:6,} ({data['percentage']:5.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"Saved processed data to: {output_dir}")
    logger.info(f"Saved statistics to: {stats_file}")

def main():
    """Main context building pipeline"""
    config = load_config()
    global logger
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("STEP 2: BUILD CONTEXTS")
    logger.info("="*80)
    
    # Load cleaned data
    products, reviews = load_cleaned_data(config)
    
    # Create contexts
    reviews, context_mapping = create_contexts(reviews, config)
    
    # Analyze context distribution
    context_stats = analyze_context_distribution(reviews, context_mapping, config)
    
    # Process product attributes
    products = process_product_attributes(products, config)
    
    # Calculate context-attribute distributions
    context_distributions = calculate_context_attribute_distributions(
        reviews, products, context_mapping, config
    )
    
    # Create visualizations
    create_context_attribute_heatmap(context_distributions, config)
    
    # Save processed data
    save_processed_data(
        reviews, products, context_mapping, context_stats, 
        context_distributions, config
    )
    
    logger.info("="*80)
    logger.info("CONTEXT BUILDING COMPLETE!")
    logger.info("="*80)
    
    return reviews, products, context_mapping, context_distributions

if __name__ == "__main__":
    main()
