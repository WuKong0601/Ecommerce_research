"""
Step 4: User Segmentation for CoFARS-Sparse
Segment users into power/regular/cold-start based on interaction count

This script:
1. Loads reviews with contexts
2. Counts interactions per user
3. Segments users into 3 groups
4. Analyzes segment statistics
5. Saves segmentation results for hybrid modeling
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.logger import setup_logger

def load_reviews_with_contexts(config):
    """Load reviews with context labels"""
    logger.info("Loading reviews with contexts...")
    
    contexts_dir = os.path.join(config['data']['processed_dir'], 'contexts')
    reviews_path = os.path.join(contexts_dir, 'reviews_with_contexts.csv')
    
    reviews = pd.read_csv(reviews_path)
    logger.info(f"Loaded {len(reviews):,} reviews")
    
    return reviews

def segment_users(reviews, config):
    """
    Segment users into 3 groups based on interaction count
    
    Returns:
        user_segments: DataFrame with user_id, segment, interaction_count
    """
    logger.info("Segmenting users...")
    
    # Count interactions per user
    user_interaction_counts = reviews.groupby('customer_id').size().reset_index()
    user_interaction_counts.columns = ['customer_id', 'interaction_count']
    
    # Define segments
    def classify_user(count):
        if count >= 5:
            return 'power'
        elif count >= 2:
            return 'regular'
        else:
            return 'cold_start'
    
    user_interaction_counts['segment'] = user_interaction_counts['interaction_count'].apply(classify_user)
    
    # Statistics
    segment_stats = user_interaction_counts['segment'].value_counts()
    
    logger.info("User segmentation results:")
    logger.info(f"  Power users (≥5 interactions): {segment_stats.get('power', 0):,}")
    logger.info(f"  Regular users (2-4 interactions): {segment_stats.get('regular', 0):,}")
    logger.info(f"  Cold-start users (1 interaction): {segment_stats.get('cold_start', 0):,}")
    
    return user_interaction_counts

def analyze_segments(user_segments, reviews):
    """Detailed analysis of user segments"""
    logger.info("Analyzing segment characteristics...")
    
    stats = {}
    
    for segment in ['power', 'regular', 'cold_start']:
        segment_users = user_segments[user_segments['segment'] == segment]['customer_id']
        segment_reviews = reviews[reviews['customer_id'].isin(segment_users)]
        
        stats[segment] = {
            'num_users': len(segment_users),
            'num_reviews': len(segment_reviews),
            'avg_interactions': segment_reviews.groupby('customer_id').size().mean(),
            'total_products': segment_reviews['product_id'].nunique(),
            'avg_products_per_user': segment_reviews.groupby('customer_id')['product_id'].nunique().mean(),
            'contexts_covered': segment_reviews['context'].nunique(),
            'avg_rating': segment_reviews['rating'].mean()
        }
        
        logger.info(f"\n{segment.upper()} SEGMENT:")
        logger.info(f"  Users: {stats[segment]['num_users']:,}")
        logger.info(f"  Reviews: {stats[segment]['num_reviews']:,} ({stats[segment]['num_reviews']/len(reviews)*100:.1f}%)")
        logger.info(f"  Avg interactions/user: {stats[segment]['avg_interactions']:.2f}")
        logger.info(f"  Avg products/user: {stats[segment]['avg_products_per_user']:.2f}")
        logger.info(f"  Contexts covered: {stats[segment]['contexts_covered']}/10")
        logger.info(f"  Avg rating: {stats[segment]['avg_rating']:.2f}")
    
    return stats

def create_segment_visualizations(user_segments, segment_stats, config):
    """Create visualizations for user segments"""
    logger.info("Creating segment visualizations...")
    
    figures_dir = config['results']['figures_dir']
    
    # Figure 1: Segment distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pie chart: User count by segment
    ax1 = axes[0, 0]
    segment_counts = user_segments['segment'].value_counts()
    ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('User Distribution by Segment')
    
    # Bar chart: Review count by segment
    ax2 = axes[0, 1]
    review_counts = []
    for seg in ['power', 'regular', 'cold_start']:
        review_counts.append(segment_stats[seg]['num_reviews'])
    ax2.bar(['Power', 'Regular', 'Cold Start'], review_counts, color=['green', 'orange', 'red'])
    ax2.set_ylabel('Number of Reviews')
    ax2.set_title('Review Distribution by Segment')
    ax2.grid(axis='y', alpha=0.3)
    
    # Histogram: Interaction count distribution
    ax3 = axes[1, 0]
    ax3.hist(user_segments['interaction_count'], bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=2, color='orange', linestyle='--', label='Regular threshold')
    ax3.axvline(x=5, color='green', linestyle='--', label='Power threshold')
    ax3.set_xlabel('Interaction Count')
    ax3.set_ylabel('Number of Users')
    ax3.set_title('User Interaction Distribution')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Box plot: Interaction count by segment
    ax4 = axes[1, 1]
    segment_data = [
        user_segments[user_segments['segment'] == 'power']['interaction_count'],
        user_segments[user_segments['segment'] == 'regular']['interaction_count'],
        user_segments[user_segments['segment'] == 'cold_start']['interaction_count']
    ]
    ax4.boxplot(segment_data, labels=['Power', 'Regular', 'Cold Start'])
    ax4.set_ylabel('Interaction Count')
    ax4.set_title('Interaction Distribution by Segment')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'user_segmentation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved segmentation visualization to: {save_path}")

def save_segmentation_results(user_segments, segment_stats, config):
    """Save segmentation results"""
    logger.info("Saving segmentation results...")
    
    # Create output directory
    output_dir = os.path.join(config['data']['processed_dir'], 'segmentation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save user segments
    segments_path = os.path.join(output_dir, 'user_segments.csv')
    user_segments.to_csv(segments_path, index=False)
    logger.info(f"Saved user segments to: {segments_path}")
    
    # Save statistics
    stats_dir = config['results']['statistics_dir']
    stats_file = os.path.join(stats_dir, 'user_segmentation_stats.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("USER SEGMENTATION STATISTICS (CoFARS-Sparse)\n")
        f.write("="*80 + "\n\n")
        
        f.write("SEGMENTATION STRATEGY:\n")
        f.write("-"*80 + "\n")
        f.write("  Power users: ≥5 interactions (full CoFARS)\n")
        f.write("  Regular users: 2-4 interactions (hybrid approach)\n")
        f.write("  Cold-start users: 1 interaction (context-based)\n\n")
        
        for segment in ['power', 'regular', 'cold_start']:
            f.write(f"\n{segment.upper()} SEGMENT:\n")
            f.write("-"*80 + "\n")
            for key, value in segment_stats[segment].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value:,}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("HYBRID MODELING STRATEGY:\n")
        f.write("="*80 + "\n")
        total_users = sum(s['num_users'] for s in segment_stats.values())
        total_reviews = sum(s['num_reviews'] for s in segment_stats.values())
        
        f.write(f"  Total users: {total_users:,}\n")
        f.write(f"  Total reviews: {total_reviews:,}\n\n")
        
        for segment in ['power', 'regular', 'cold_start']:
            pct_users = segment_stats[segment]['num_users'] / total_users * 100
            pct_reviews = segment_stats[segment]['num_reviews'] / total_reviews * 100
            f.write(f"  {segment.capitalize()}: {pct_users:.1f}% users, {pct_reviews:.1f}% reviews\n")
    
    logger.info(f"Saved statistics to: {stats_file}")
    
    return segments_path

def main():
    """Main user segmentation pipeline"""
    config = load_config()
    global logger
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("STEP 4: USER SEGMENTATION (CoFARS-Sparse)")
    logger.info("="*80)
    
    # Load reviews
    reviews = load_reviews_with_contexts(config)
    
    # Segment users
    user_segments = segment_users(reviews, config)
    
    # Analyze segments
    segment_stats = analyze_segments(user_segments, reviews)
    
    # Create visualizations
    create_segment_visualizations(user_segments, segment_stats, config)
    
    # Save results
    segments_path = save_segmentation_results(user_segments, segment_stats, config)
    
    logger.info("="*80)
    logger.info("USER SEGMENTATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Segments saved to: {segments_path}")
    
    return user_segments, segment_stats

if __name__ == "__main__":
    main()
