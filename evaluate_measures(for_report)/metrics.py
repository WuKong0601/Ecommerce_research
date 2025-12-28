"""
Comprehensive Evaluation Metrics for Recommendation Systems
Dùng cho báo cáo môn học - So sánh với các thuật toán khác

Metrics implemented:
- Precision@K
- Recall@K  
- F1@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- NMAE (Normalized Mean Absolute Error)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RecommendationMetrics:
    """Class tính toán các metrics đánh giá hệ thống gợi ý"""
    
    def __init__(self):
        pass
    
    # ==================== RANKING METRICS ====================
    
    @staticmethod
    def precision_at_k(y_true, y_pred, k=10):
        """
        Precision@K: Tỷ lệ items đúng trong top-K recommendations
        
        Args:
            y_true: List of relevant items (ground truth)
            y_pred: List of recommended items (ranked)
            k: Number of top items to consider
            
        Returns:
            Precision@K score (0-1)
            
        Example:
            y_true = [1, 3, 5, 7]
            y_pred = [1, 2, 3, 4, 5]  # Top 5 recommendations
            precision_at_k(y_true, y_pred, k=5) = 3/5 = 0.6
        """
        if len(y_pred) == 0:
            return 0.0
        
        # Lấy top-K predictions
        top_k = y_pred[:k]
        
        # Đếm số items đúng trong top-K
        hits = len(set(top_k) & set(y_true))
        
        # Precision = hits / k
        return hits / min(k, len(top_k))
    
    @staticmethod
    def recall_at_k(y_true, y_pred, k=10):
        """
        Recall@K: Tỷ lệ items đúng được tìm thấy trong top-K
        
        Args:
            y_true: List of relevant items (ground truth)
            y_pred: List of recommended items (ranked)
            k: Number of top items to consider
            
        Returns:
            Recall@K score (0-1)
            
        Example:
            y_true = [1, 3, 5, 7]  # 4 relevant items
            y_pred = [1, 2, 3, 4, 5]
            recall_at_k(y_true, y_pred, k=5) = 3/4 = 0.75
        """
        if len(y_true) == 0:
            return 0.0
        
        # Lấy top-K predictions
        top_k = y_pred[:k]
        
        # Đếm số items đúng trong top-K
        hits = len(set(top_k) & set(y_true))
        
        # Recall = hits / total_relevant
        return hits / len(y_true)
    
    @staticmethod
    def f1_at_k(y_true, y_pred, k=10):
        """
        F1@K: Harmonic mean of Precision@K and Recall@K
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            y_true: List of relevant items
            y_pred: List of recommended items
            k: Number of top items
            
        Returns:
            F1@K score (0-1)
        """
        precision = RecommendationMetrics.precision_at_k(y_true, y_pred, k)
        recall = RecommendationMetrics.recall_at_k(y_true, y_pred, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(y_true, y_pred):
        """
        MRR: Mean Reciprocal Rank
        
        Đo vị trí của item đúng đầu tiên trong danh sách gợi ý
        MRR = 1 / rank_of_first_relevant_item
        
        Args:
            y_true: List of relevant items
            y_pred: List of recommended items (ranked)
            
        Returns:
            MRR score
            
        Example:
            y_true = [3, 5, 7]
            y_pred = [1, 2, 3, 4, 5]  # Item 3 ở vị trí 3
            MRR = 1/3 = 0.333
        """
        for i, item in enumerate(y_pred, 1):
            if item in y_true:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(y_true, y_pred, k=10, relevance_scores=None):
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Đo chất lượng ranking, items đúng ở vị trí cao được score cao hơn
        
        DCG = sum(rel_i / log2(i+1)) for i in 1..k
        IDCG = DCG của ideal ranking
        NDCG = DCG / IDCG
        
        Args:
            y_true: List of relevant items
            y_pred: List of recommended items (ranked)
            k: Number of top items
            relevance_scores: Dict {item_id: relevance_score}
                             Nếu None, dùng binary relevance (1 nếu relevant, 0 nếu không)
            
        Returns:
            NDCG@K score (0-1)
        """
        # Lấy top-K predictions
        top_k = y_pred[:k]
        
        # Tính DCG
        dcg = 0.0
        for i, item in enumerate(top_k, 1):
            if relevance_scores is not None:
                rel = relevance_scores.get(item, 0)
            else:
                # Binary relevance
                rel = 1.0 if item in y_true else 0.0
            
            # DCG formula: rel / log2(i+1)
            dcg += rel / np.log2(i + 1)
        
        # Tính IDCG (Ideal DCG)
        if relevance_scores is not None:
            # Sort by relevance scores descending
            ideal_items = sorted(relevance_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:k]
            ideal_rels = [rel for _, rel in ideal_items]
        else:
            # Binary: ideal là tất cả relevant items ở top
            ideal_rels = [1.0] * min(len(y_true), k)
        
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_rels, 1))
        
        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    # ==================== RATING PREDICTION METRICS ====================
    
    @staticmethod
    def rmse(y_true, y_pred):
        """
        RMSE: Root Mean Squared Error
        
        RMSE = sqrt(mean((y_true - y_pred)^2))
        
        Args:
            y_true: Array of true ratings
            y_pred: Array of predicted ratings
            
        Returns:
            RMSE value (lower is better)
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        MAE: Mean Absolute Error
        
        MAE = mean(|y_true - y_pred|)
        
        Args:
            y_true: Array of true ratings
            y_pred: Array of predicted ratings
            
        Returns:
            MAE value (lower is better)
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def nmae(y_true, y_pred, rating_range=5.0):
        """
        NMAE: Normalized Mean Absolute Error
        
        NMAE = MAE / rating_range
        
        Normalize MAE by rating range (e.g., 1-5 stars → range = 4)
        
        Args:
            y_true: Array of true ratings
            y_pred: Array of predicted ratings
            rating_range: Range of rating scale (default: 5 for 1-5 stars)
            
        Returns:
            NMAE value (0-1, lower is better)
        """
        mae_value = mean_absolute_error(y_true, y_pred)
        return mae_value / rating_range
    
    # ==================== BATCH EVALUATION ====================
    
    @staticmethod
    def evaluate_all_ranking_metrics(y_true_list, y_pred_list, k_values=[5, 10, 20]):
        """
        Đánh giá tất cả ranking metrics cho nhiều users
        
        Args:
            y_true_list: List of ground truth item lists [[items_user1], [items_user2], ...]
            y_pred_list: List of predicted item lists [[preds_user1], [preds_user2], ...]
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Evaluate for each K
        for k in k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            
            for y_true, y_pred in zip(y_true_list, y_pred_list):
                precision_scores.append(
                    RecommendationMetrics.precision_at_k(y_true, y_pred, k)
                )
                recall_scores.append(
                    RecommendationMetrics.recall_at_k(y_true, y_pred, k)
                )
                f1_scores.append(
                    RecommendationMetrics.f1_at_k(y_true, y_pred, k)
                )
                ndcg_scores.append(
                    RecommendationMetrics.ndcg_at_k(y_true, y_pred, k)
                )
            
            results[f'Precision@{k}'] = np.mean(precision_scores)
            results[f'Recall@{k}'] = np.mean(recall_scores)
            results[f'F1@{k}'] = np.mean(f1_scores)
            results[f'NDCG@{k}'] = np.mean(ndcg_scores)
        
        # MRR (không phụ thuộc K)
        mrr_scores = [
            RecommendationMetrics.mean_reciprocal_rank(y_true, y_pred)
            for y_true, y_pred in zip(y_true_list, y_pred_list)
        ]
        results['MRR'] = np.mean(mrr_scores)
        
        return results
    
    @staticmethod
    def evaluate_all_rating_metrics(y_true, y_pred):
        """
        Đánh giá tất cả rating prediction metrics
        
        Args:
            y_true: Array of true ratings
            y_pred: Array of predicted ratings
            
        Returns:
            Dictionary of metrics
        """
        return {
            'RMSE': RecommendationMetrics.rmse(y_true, y_pred),
            'MAE': RecommendationMetrics.mae(y_true, y_pred),
            'NMAE': RecommendationMetrics.nmae(y_true, y_pred, rating_range=4.0)  # 1-5 scale
        }


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("="*80)
    print("DEMONSTRATION: Recommendation Metrics")
    print("="*80)
    
    # Example 1: Single user ranking metrics
    print("\n--- Example 1: Ranking Metrics (Single User) ---")
    y_true = [3, 5, 7, 9]  # Ground truth relevant items
    y_pred = [1, 3, 2, 5, 4, 7, 6]  # Model predictions (ranked)
    
    k = 5
    print(f"Ground truth: {y_true}")
    print(f"Predictions (top {k}): {y_pred[:k]}")
    print(f"\nPrecision@{k}: {RecommendationMetrics.precision_at_k(y_true, y_pred, k):.4f}")
    print(f"Recall@{k}: {RecommendationMetrics.recall_at_k(y_true, y_pred, k):.4f}")
    print(f"F1@{k}: {RecommendationMetrics.f1_at_k(y_true, y_pred, k):.4f}")
    print(f"NDCG@{k}: {RecommendationMetrics.ndcg_at_k(y_true, y_pred, k):.4f}")
    print(f"MRR: {RecommendationMetrics.mean_reciprocal_rank(y_true, y_pred):.4f}")
    
    # Example 2: Multiple users
    print("\n--- Example 2: Batch Evaluation (Multiple Users) ---")
    y_true_list = [
        [3, 5, 7],
        [1, 4, 8],
        [2, 6, 9, 10]
    ]
    y_pred_list = [
        [3, 1, 5, 2, 7],
        [2, 4, 1, 3, 5],
        [6, 2, 9, 1, 10]
    ]
    
    results = RecommendationMetrics.evaluate_all_ranking_metrics(
        y_true_list, y_pred_list, k_values=[3, 5]
    )
    
    print("Results across all users:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Example 3: Rating prediction metrics
    print("\n--- Example 3: Rating Prediction Metrics ---")
    y_true_ratings = np.array([5, 4, 3, 5, 2, 4, 3, 5, 4])
    y_pred_ratings = np.array([4.5, 4.2, 3.1, 4.8, 2.3, 3.9, 3.2, 4.7, 4.1])
    
    print(f"True ratings: {y_true_ratings}")
    print(f"Pred ratings: {y_pred_ratings}")
    
    rating_results = RecommendationMetrics.evaluate_all_rating_metrics(
        y_true_ratings, y_pred_ratings
    )
    
    print("\nRating Prediction Metrics:")
    for metric, value in rating_results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*80)
    print("Metrics module ready to use!")
    print("="*80)
