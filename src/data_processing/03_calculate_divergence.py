"""
Step 3: Calculate JS Divergence between Contexts
Implements KL and JS divergence calculation for context similarity

This script:
1. Loads context distributions
2. Implements KL divergence function
3. Implements JS divergence function  
4. Calculates pairwise JS divergence matrix
5. Creates similarity visualization (heatmap like Figure 5 in paper)
6. Saves results for model training
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.logger import setup_logger

def load_context_distributions(config):
    """Load context distributions from previous step"""
    logger.info("Loading context distributions...")
    
    contexts_dir = os.path.join(config['data']['processed_dir'], 'contexts')
    dist_path = os.path.join(contexts_dir, 'context_distributions.json')
    
    with open(dist_path, 'r') as f:
        context_distributions = json.load(f)
    
    logger.info(f"Loaded distributions for {len(context_distributions)} contexts")
    
    return context_distributions

def normalize_distribution(dist_dict, smoothing=1e-8):
    """
    Normalize a distribution dictionary and add smoothing
    
    Args:
        dist_dict: Dictionary of {value: probability}
        smoothing: Small value to add to avoid log(0)
        
    Returns:
        Normalized dictionary with smoothing
    """
    # Get all values and add smoothing
    values = np.array(list(dist_dict.values())) + smoothing
    
    # Normalize
    total = values.sum()
    normalized = values / total
    
    # Create new dict
    result = {key: normalized[i] for i, key in enumerate(dist_dict.keys())}
    
    return result

def merge_distributions(distributions_list, all_keys):
    """
    Merge multiple distribution dictionaries into single distribution vector
    
    Args:
        distributions_list: List of distribution dictionaries
        all_keys: All possible keys across all distributions
        
    Returns:
        Merged probability vector
    """
    merged = []
    
    for dist_dict in distributions_list:
        for key in sorted(all_keys):  # Sort for consistency
            value = float(dist_dict.get(key, 0))  # Ensure float type
            merged.append(value)
    
    return np.array(merged, dtype=np.float64)  # Explicit dtype

def calculate_kl_divergence(P, Q, smoothing=1e-8):
    """
    Calculate KL divergence KL(P||Q) = sum(P * log(P/Q))
    
    Args:
        P: Probability distribution (numpy array)
        Q: Probability distribution (numpy array)
        smoothing: Small value to avoid division by zero
        
    Returns:
        KL divergence value
    """
    # Add smoothing
    P = P + smoothing
    Q = Q + smoothing
    
    # Normalize
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Calculate KL divergence
    kl = np.sum(P * np.log(P / Q))
    
    return kl

def calculate_js_divergence(P, Q, smoothing=1e-8):
    """
    Calculate Jensen-Shannon divergence
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Args:
        P: Probability distribution (numpy array)
        Q: Probability distribution (numpy array)
        smoothing: Small value to avoid division by zero
        
    Returns:
        JS divergence value (symmetric, bounded between 0 and 1)
    """
    # Add smoothing
    P = P + smoothing
    Q = Q + smoothing
    
    # Normalize
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Calculate midpoint distribution
    M = 0.5 * (P + Q)
    
    # Calculate JS divergence
    js = 0.5 * calculate_kl_divergence(P, M, smoothing=0) + \
         0.5 * calculate_kl_divergence(Q, M, smoothing=0)
    
    return js

def get_all_attribute_values(context_distributions):
    """Get all unique values for each attribute"""
    all_categories = set()
    all_price_buckets = set()
    all_rating_levels = set()
    
    for ctx_data in context_distributions.values():
        all_categories.update(ctx_data['distributions']['category'].keys())
        all_price_buckets.update(ctx_data['distributions']['price_bucket'].keys())
        all_rating_levels.update(ctx_data['distributions']['rating_level'].keys())
    
    return (sorted(list(all_categories)), 
            sorted(list(all_price_buckets)),
            sorted(list(all_rating_levels)))

def get_context_distribution_vector(context_data, all_categories, all_price_buckets, 
                                     all_rating_levels, smoothing=1e-8):
    """
    Convert context distribution to unified vector for divergence calculation
    
    Args:
        context_data: Context distribution data
        all_categories: All possible categories (sorted list)
        all_price_buckets: All possible price buckets (sorted list)
        all_rating_levels: All possible rating levels (sorted list)
        smoothing: Smoothing parameter
        
    Returns:
        Probability vector representing the context
    """
    distributions = context_data['distributions']
    
    # Build vector for each attribute
    all_probs = []
    
    # Add category probabilities
    for cat in all_categories:
        prob = distributions['category'].get(cat, 0.0)
        all_probs.append(float(prob))
    
    # Add price bucket probabilities
    for price in all_price_buckets:
        prob = distributions['price_bucket'].get(price, 0.0)
        all_probs.append(float(prob))
    
    # Add rating level probabilities
    for rating in all_rating_levels:
        prob = distributions['rating_level'].get(rating, 0.0)
        all_probs.append(float(prob))
    
    # Convert to numpy array
    full_vector = np.array(all_probs, dtype=np.float64)
    
    # Add smoothing
    full_vector = full_vector + float(smoothing)
    
    # Normalize
    full_vector = full_vector / full_vector.sum()
    
    return full_vector

def calculate_all_pairwise_js_divergences(context_distributions, config):
    """
    Calculate pairwise JS divergence between all contexts
    
    Returns:
        js_matrix: Dictionary of {(context_i, context_j): js_divergence}
        similarity_matrix: Dictionary of {(context_i, context_j): similarity}
            where similarity = 1 - js_divergence
    """
    logger.info("Calculating pairwise JS divergences...")
    
    smoothing = config['divergence']['smoothing']
    
    # Get all unique values for each attribute
    all_categories, all_price_buckets, all_rating_levels = get_all_attribute_values(
        context_distributions
    )
    
    logger.info(f"Attribute vocabulary sizes:")
    logger.info(f"  Categories: {len(all_categories)}")
    logger.info(f"  Price buckets: {len(all_price_buckets)}")
    logger.info(f"  Rating levels: {len(all_rating_levels)}")
    
    # Convert all contexts to vectors
    context_vectors = {}
    contexts = sorted(context_distributions.keys())
    
    for context in contexts:
        vector = get_context_distribution_vector(
            context_distributions[context],
            all_categories,
            all_price_buckets,
            all_rating_levels,
            smoothing
        )
        context_vectors[context] = vector
    
    # Calculate pairwise JS divergences
    js_matrix = {}
    similarity_matrix = {}
    
    for i, ctx_i in enumerate(contexts):
        for j, ctx_j in enumerate(contexts):
            if i <= j:  # Only calculate upper triangle (symmetric)
                js = calculate_js_divergence(
                    context_vectors[ctx_i],
                    context_vectors[ctx_j],
                    smoothing=0  # Already smoothed in vector creation
                )
                
                js_matrix[(ctx_i, ctx_j)] = float(js)
                js_matrix[(ctx_j, ctx_i)] = float(js)  # Symmetric
                
                similarity = 1 - js
                similarity_matrix[(ctx_i, ctx_j)] = float(similarity)
                similarity_matrix[(ctx_j, ctx_i)] = float(similarity)
    
    logger.info(f"Calculated {len(js_matrix)} pairwise divergences")
    
    # Log some examples
    logger.info("\nExample JS divergences:")
    sample_pairs = list(js_matrix.items())[:5]
    for (ctx_i, ctx_j), js in sample_pairs:
        logger.info(f"  JS({ctx_i}, {ctx_j}) = {js:.4f}, Similarity = {1-js:.4f}")
    
    return js_matrix, similarity_matrix, contexts

def create_similarity_heatmap(similarity_matrix, contexts, config):
    """
    Create similarity heatmap visualization (like Figure 5 in paper)
    """
    logger.info("Creating similarity heatmap...")
    
    # Create matrix for heatmap
    n = len(contexts)
    matrix = np.zeros((n, n))
    
    for i, ctx_i in enumerate(contexts):
        for j, ctx_j in enumerate(contexts):
            matrix[i, j] = similarity_matrix[(ctx_i, ctx_j)]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        matrix,
        xticklabels=contexts,
        yticklabels=contexts,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Similarity (1 - JS Divergence)'},
        ax=ax
    )
    
    ax.set_xlabel('Context')
    ax.set_ylabel('Context')
    ax.set_title('Context Similarity Matrix (Based on JS Divergence)')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    figures_dir = config['results']['figures_dir']
    save_path = os.path.join(figures_dir, 'context_similarity_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity heatmap to: {save_path}")

def analyze_similarity_patterns(similarity_matrix, contexts):
    """Analyze interesting patterns in context similarities"""
    logger.info("Analyzing similarity patterns...")
    
    stats = {}
    
    # Find most similar context pairs (excluding diagonal)
    similarities = []
    for (ctx_i, ctx_j), sim in similarity_matrix.items():
        if ctx_i != ctx_j:
            similarities.append(((ctx_i, ctx_j), sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    stats['most_similar_pairs'] = similarities[:5]
    stats['least_similar_pairs'] = similarities[-5:]
    
    logger.info("\nMost similar context pairs:")
    for (ctx_i, ctx_j), sim in stats['most_similar_pairs']:
        logger.info(f"  {ctx_i} <-> {ctx_j}: {sim:.4f}")
    
    logger.info("\nLeast similar context pairs:")
    for (ctx_i, ctx_j), sim in stats['least_similar_pairs']:
        logger.info(f"  {ctx_i} <-> {ctx_j}: {sim:.4f}")
    
    return stats

def save_divergence_results(js_matrix, similarity_matrix, contexts, similarity_stats, config):
    """Save all divergence results"""
    logger.info("Saving divergence results...")
    
    # Create output directory
    output_dir = os.path.join(config['data']['processed_dir'], 'divergences')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays for model training
    n = len(contexts)
    js_array = np.zeros((n, n))
    sim_array = np.zeros((n, n))
    
    for i, ctx_i in enumerate(contexts):
        for j, ctx_j in enumerate(contexts):
            js_array[i, j] = js_matrix[(ctx_i, ctx_j)]
            sim_array[i, j] = similarity_matrix[(ctx_i, ctx_j)]
    
    np.save(os.path.join(output_dir, 'js_divergence_matrix.npy'), js_array)
    np.save(os.path.join(output_dir, 'similarity_matrix.npy'), sim_array)
    
    # Save context order
    with open(os.path.join(output_dir, 'context_order.json'), 'w') as f:
        json.dump(contexts, f, indent=2)
    
    # Save dictionary version for reference
    js_dict = {f"{ctx_i}_{ctx_j}": float(js) 
               for (ctx_i, ctx_j), js in js_matrix.items()}
    with open(os.path.join(output_dir, 'js_divergences.json'), 'w') as f:
        json.dump(js_dict, f, indent=2)
    
    # Save statistics
    stats_dir = config['results']['statistics_dir']
    stats_file = os.path.join(stats_dir, 'js_divergence_stats.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("JS DIVERGENCE STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total contexts: {len(contexts)}\n")
        f.write(f"Total pairwise comparisons: {len(js_matrix)}\n\n")
        
        f.write("JS Divergence Distribution:\n")
        f.write(f"  Mean: {np.mean(list(js_matrix.values())):.4f}\n")
        f.write(f"  Std: {np.std(list(js_matrix.values())):.4f}\n")
        f.write(f"  Min: {np.min(list(js_matrix.values())):.4f}\n")
        f.write(f"  Max: {np.max(list(js_matrix.values())):.4f}\n\n")
        
        f.write("Most Similar Context Pairs:\n")
        f.write("-"*80 + "\n")
        for (ctx_i, ctx_j), sim in similarity_stats['most_similar_pairs']:
            f.write(f"  {ctx_i:25s} <-> {ctx_j:25s}: {sim:.4f}\n")
        
        f.write("\nLeast Similar Context Pairs:\n")
        f.write("-"*80 + "\n")
        for (ctx_i, ctx_j), sim in similarity_stats['least_similar_pairs']:
            f.write(f"  {ctx_i:25s} <-> {ctx_j:25s}: {sim:.4f}\n")
    
    logger.info(f"Saved divergence results to: {output_dir}")
    logger.info(f"Saved statistics to: {stats_file}")

def main():
    """Main JS divergence calculation pipeline"""
    config = load_config()
    global logger
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("STEP 3: CALCULATE JS DIVERGENCE")
    logger.info("="*80)
    
    # Load context distributions
    context_distributions = load_context_distributions(config)
    
    # Calculate pairwise JS divergences
    js_matrix, similarity_matrix, contexts = calculate_all_pairwise_js_divergences(
        context_distributions, config
    )
    
    # Create visualizations
    create_similarity_heatmap(similarity_matrix, contexts, config)
    
    # Analyze patterns
    similarity_stats = analyze_similarity_patterns(similarity_matrix, contexts)
    
    # Save results
    save_divergence_results(js_matrix, similarity_matrix, contexts, similarity_stats, config)
    
    logger.info("="*80)
    logger.info("JS DIVERGENCE CALCULATION COMPLETE!")
    logger.info("="*80)
    
    return js_matrix, similarity_matrix

if __name__ == "__main__":
    main()
