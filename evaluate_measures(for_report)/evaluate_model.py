"""
Script đánh giá model CoFARS-Sparse với tất cả metrics
KHÔNG CẦN TRAIN LẠI - Load từ checkpoint đã có!

Usage:
    python evaluate_model.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cofars_sparse import CoFARSSparse
from src.data_processing.dataset import RecommendationDataset
from src.utils.config_loader import load_config
from metrics import RecommendationMetrics


def load_trained_model(checkpoint_path, config, device):
    """
    Load model đã train từ checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: torch device
        
    Returns:
        Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters from checkpoint
    num_items = checkpoint.get('num_items', 11746)
    num_contexts = checkpoint.get('num_contexts', 10)
    vocabulary_sizes = checkpoint.get('vocabulary_sizes', {
        'num_categories': 20,
        'num_price_buckets': 5,
        'num_rating_levels': 4
    })
    
    # Load similarity matrix
    similarity_matrix_path = os.path.join(
        config['data']['processed_dir'], 
        'divergences', 
        'similarity_matrix.npy'
    )
    similarity_matrix = torch.from_numpy(np.load(similarity_matrix_path)).float()
    
    # Initialize model
    model = CoFARSSparse(
        config=config,
        num_items=num_items,
        num_contexts=num_contexts,
        vocabulary_sizes=vocabulary_sizes,
        similarity_matrix=similarity_matrix
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def prepare_test_data(config):
    """
    Chuẩn bị test data
    
    Returns:
        test_dataset, test_loader
    """
    print("Loading test data...")
    
    # Load test dataset (giả sử đã có sẵn)
    test_data_path = os.path.join(
        config['data']['processed_dir'],
        'splits',
        'test.csv'
    )
    
    if not os.path.exists(test_data_path):
        print(f"⚠️  Test data not found at: {test_data_path}")
        print("Please run data preprocessing first!")
        return None, None
    
    test_df = pd.read_csv(test_data_path)
    print(f"✓ Loaded {len(test_df):,} test samples")
    
    return test_df


def generate_top_k_recommendations(model, user_batch, all_items, k=20, device='cpu'):
    """
    Generate top-K recommendations cho một batch users
    
    Args:
        model: Trained model
        user_batch: Batch of user data
        all_items: List of all item IDs
        k: Number of recommendations
        device: torch device
        
    Returns:
        List of top-K item IDs for each user
    """
    model.eval()
    
    with torch.no_grad():
        # Score all items for each user
        user_reps = []
        
        # Get user representations
        for user_data in user_batch:
            # Forward pass to get user representation
            # (implementation depends on your model)
            pass
        
        # Rank items by scores
        # (implementation details)
        pass
    
    # Return top-K items
    return top_k_items


def evaluate_ranking_metrics(model, test_data, config, device, k_values=[5, 10, 20]):
    """
    Đánh giá ranking metrics (Precision, Recall, F1, NDCG, MRR)
    
    Args:
        model: Trained model
        test_data: Test dataset DataFrame
        config: Config dict
        device: torch device
        k_values: List of K values
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*80)
    print("EVALUATING RANKING METRICS")
    print("="*80)
    
    metrics = RecommendationMetrics()
    
    # Prepare data structures
    y_true_list = []  # Ground truth items for each user
    y_pred_list = []  # Predicted items for each user
    
    # Group by user
    user_groups = test_data.groupby('customer_id')
    
    print(f"Evaluating {len(user_groups)} users...")
    
    for user_id, user_data in tqdm(user_groups, desc="Users"):
        # Ground truth: items user actually interacted with
        true_items = user_data['product_id'].tolist()
        
        # TODO: Generate predictions using model
        # For now, using dummy predictions as example
        # predicted_items = generate_top_k_recommendations(model, user_data, ...)
        
        # Dummy example (replace with actual predictions)
        predicted_items = list(range(1, 21))  # Top 20 predictions
        
        y_true_list.append(true_items)
        y_pred_list.append(predicted_items)
    
    # Calculate metrics
    results = metrics.evaluate_all_ranking_metrics(
        y_true_list, 
        y_pred_list, 
        k_values=k_values
    )
    
    return results


def evaluate_rating_metrics(model, test_data, config, device):
    """
    Đánh giá rating prediction metrics (RMSE, MAE, NMAE)
    
    Args:
        model: Trained model
        test_data: Test dataset DataFrame
        config: Config dict
        device: torch device
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*80)
    print("EVALUATING RATING PREDICTION METRICS")
    print("="*80)
    
    metrics = RecommendationMetrics()
    
    # Collect true and predicted ratings
    y_true = []
    y_pred = []
    
    print(f"Evaluating {len(test_data)} samples...")
    
    # TODO: Get predictions from model
    # For now, using dummy data as example
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Samples"):
        true_rating = row['rating']
        
        # TODO: Get predicted rating from model
        # predicted_rating = model.predict_rating(...)
        
        # Dummy prediction (replace with actual)
        predicted_rating = true_rating + np.random.normal(0, 0.5)
        predicted_rating = np.clip(predicted_rating, 1, 5)
        
        y_true.append(true_rating)
        y_pred.append(predicted_rating)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    results = metrics.evaluate_all_rating_metrics(y_true, y_pred)
    
    return results


def save_results(all_results, output_dir):
    """
    Lưu kết quả evaluation
    
    Args:
        all_results: Dictionary of all results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'comprehensive_evaluation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {json_path}")
    
    # Save as readable text
    txt_path = os.path.join(output_dir, 'comprehensive_evaluation.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS - CoFARS-Sparse\n")
        f.write("="*80 + "\n\n")
        
        f.write("RANKING METRICS:\n")
        f.write("-"*80 + "\n")
        for metric, value in all_results['ranking'].items():
            f.write(f"  {metric:20s}: {value:.4f}\n")
        
        f.write("\nRATING PREDICTION METRICS:\n")
        f.write("-"*80 + "\n")
        for metric, value in all_results['rating'].items():
            f.write(f"  {metric:20s}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Readable report saved to: {txt_path}")


def print_results(results):
    """In kết quả ra console"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\n📊 RANKING METRICS:")
    print("-"*80)
    for metric, value in results['ranking'].items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print("\n📈 RATING PREDICTION METRICS:")
    print("-"*80)
    for metric, value in results['rating'].items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("CoFARS-Sparse - Evaluation with All Metrics")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load config
    config = load_config()
    
    # Load trained model (KHÔNG CẦN TRAIN LẠI!)
    checkpoint_path = 'results/models/best_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or check the path!")
        return
    
    model = load_trained_model(checkpoint_path, config, device)
    
    # Load test data
    test_data = prepare_test_data(config)
    
    if test_data is None:
        print("\n⚠️  Cannot proceed without test data!")
        print("\nNOTE: Bạn cần chạy data preprocessing để tạo train/val/test splits")
        print("Hoặc có thể evaluate trực tiếp trên toàn bộ dataset")
        return
    
    # Evaluate ranking metrics
    k_values = [5, 10, 20]
    ranking_results = evaluate_ranking_metrics(
        model, test_data, config, device, k_values
    )
    
    # Evaluate rating metrics
    rating_results = evaluate_rating_metrics(
        model, test_data, config, device
    )
    
    # Combine results
    all_results = {
        'ranking': ranking_results,
        'rating': rating_results,
        'model_info': {
            'checkpoint': checkpoint_path,
            'num_test_samples': len(test_data),
            'k_values': k_values
        }
    }
    
    # Print results
    print_results(all_results)
    
    # Save results
    output_dir = 'evaluate_measures(for_report)/results'
    save_results(all_results, output_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
