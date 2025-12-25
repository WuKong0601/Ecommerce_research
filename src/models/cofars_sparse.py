"""
Complete CoFARS-Sparse Model
Integrates all components for hybrid recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Ensure parent package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.probability_encoder import ProbabilityEncoder, JSDivergenceLoss
from src.models.prototypes import ContextPrototypes, IndependenceLoss, StaticContextMatcher
from src.models.user_modules import HybridUserEncoder

class CoFARSSparse(nn.Module):
    """
    CoFARS-Sparse: Adapted for sparse e-commerce data
    
    Key differences from original:
    1. Hybrid user modeling (3 strategies for 3 segments)
    2. Context-aggregated prototypes (not user-visit based)
    3. Static similarity matching (not temporal graph)
    4. Conditional sequence modeling (GRU only for power users)
    """
    
    def __init__(self, config, num_items, num_contexts, vocabulary_sizes, similarity_matrix):
        """
        Args:
            config: Configuration dict with hyperparameters
            num_items: Total number of products
            num_contexts: Number of distinct contexts
            vocabulary_sizes: Dict with num_categories, num_price_buckets, num_rating_levels
            similarity_matrix: (num_contexts, num_contexts) pre-computed JS similarity
        """
        super(CoFARSSparse, self).__init__()
        
        self.config = config
        self.num_items = num_items
        self.num_contexts = num_contexts
        
        # Hyperparameters
        self.embedding_dim = config['model']['embedding_dim']
        self.num_prototypes = config['model']['num_prototypes']
        self.hidden_dim = config['model'].get('gru_hidden_dim', 64)
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, self.embedding_dim)
        
        # Core components
        self.probability_encoder = ProbabilityEncoder(
            input_dim=self.embedding_dim,
            num_categories=vocabulary_sizes['num_categories'],
            num_price_buckets=vocabulary_sizes['num_price_buckets'],
            num_rating_levels=vocabulary_sizes['num_rating_levels'],
            hidden_dim=128
        )
        
        self.context_prototypes = ContextPrototypes(
            num_prototypes=self.num_prototypes,
            embedding_dim=self.embedding_dim,
            num_contexts=num_contexts
        )
        
        self.context_matcher = StaticContextMatcher(similarity_matrix)
        
        self.hybrid_encoder = HybridUserEncoder(
            item_embedding_dim=self.embedding_dim,
            context_embedding_dim=self.embedding_dim,
            output_dim=self.hidden_dim
        )
        
        # Final scoring layer
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Loss functions
        self.js_loss_fn = JSDivergenceLoss()
        self.ind_loss_fn = IndependenceLoss()
        self.rec_loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, batch):
        """
        Forward pass
        
        Args:
            batch: Dict with:
                - user_segments: (batch_size,) segment indices
                - user_item_ids: (batch_size, max_seq_len) user's item sequence
                - sequence_lengths: (batch_size,) actual lengths
                - context_ids: (batch_size,) context indices
                - candidate_item_ids: (batch_size,) items to score
                
        Returns:
            scores: (batch_size,) prediction scores
            auxiliary_outputs: Dict with intermediate results
        """
        batch_size = batch['user_segments'].shape[0]
        device = batch['user_segments'].device
        
        # 1. Get context embeddings
        context_embeddings = self.context_prototypes(batch['context_ids'])  # (batch_size, emb_dim)
        
        # 2. Enrich with similar contexts
        enriched_context = self.context_matcher.aggregate_from_similar_contexts(
            context_embeddings, 
            batch['context_ids'],
            self.context_prototypes,
            top_k=3
        )
        
        # 3. Get user item embeddings
        user_item_embeddings = self.item_embeddings(batch['user_item_ids'])  # (batch_size, seq_len, emb_dim)
        
        # 4. Hybrid user encoding
        user_representations = self.hybrid_encoder(
            batch['user_segments'],
            user_item_embeddings,
            enriched_context,
            batch.get('sequence_lengths')
        )  # (batch_size, hidden_dim)
        
        # 5. Get candidate item embeddings
        candidate_embeddings = self.item_embeddings(batch['candidate_item_ids'])  # (batch_size, emb_dim)
        
        # 6. Combine user + item for scoring
        combined = torch.cat([user_representations, candidate_embeddings], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (batch_size,)
        
        # Auxiliary outputs for loss calculation
        auxiliary = {
            'context_embeddings': context_embeddings,
            'enriched_context': enriched_context,
            'user_representations': user_representations,
            'prototypes': self.context_prototypes.prototypes
        }
        
        return scores, auxiliary
    
    def calculate_loss(self, batch, scores, auxiliary, ground_truth_js_matrix=None):
        """
        Calculate complete loss: L_REC + γ*L_MSE + λ*L_IND
        
        Args:
            batch: Input batch
            scores: (batch_size,) prediction scores
            auxiliary: Dict with intermediate outputs
            ground_truth_js_matrix: (num_contexts, num_contexts) optional
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        gamma = self.config['training'].get('gamma', 0.05)
        lambda_ind = self.config['training'].get('lambda', 0.001)
        
        # 1. Recommendation loss (L_REC)
        labels = batch['labels'].float()  # (batch_size,)
        rec_loss = self.rec_loss_fn(scores, labels)
        
        # 2. JS divergence loss (L_MSE) - optional
        js_loss = torch.tensor(0.0, device=scores.device)
        if ground_truth_js_matrix is not None and 'context_pairs' in batch:
            # Get context pairs for JS loss
            ctx_i_ids = batch['context_pairs'][:, 0]
            ctx_j_ids = batch['context_pairs'][:, 1]
            
            ctx_i_emb = self.context_prototypes(ctx_i_ids)
            ctx_j_emb = self.context_prototypes(ctx_j_ids)
            
            # Ground truth JS from matrix
            gt_js = ground_truth_js_matrix[ctx_i_ids, ctx_j_ids]
            
            js_loss, _ = self.js_loss_fn(ctx_i_emb, ctx_j_emb, self.probability_encoder, gt_js)
        
        # 3. Independence loss (L_IND)
        ind_loss = self.ind_loss_fn(auxiliary['prototypes'])
        
        # Total loss
        total_loss = rec_loss + gamma * js_loss + lambda_ind * ind_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'rec': rec_loss.item(),
            'js': js_loss.item() if isinstance(js_loss, torch.Tensor) else 0.0,
            'ind': ind_loss.item()
        }
        
        return total_loss, loss_dict
    
    def predict(self, batch):
        """
        Inference mode prediction
        
        Args:
            batch: Input batch
            
        Returns:
            probabilities: (batch_size,) prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            scores, _ = self.forward(batch)
            probabilities = torch.sigmoid(scores)
        return probabilities


if __name__ == "__main__":
    print("Testing Complete CoFARS-Sparse Model...")
    
    # Configuration
    config = {
        'model': {
            'embedding_dim': 16,
            'num_prototypes': 10,
            'gru_hidden_dim': 32,
        },
        'training': {
            'gamma': 0.05,
            'lambda': 0.001
        }
    }
    
    # Parameters
    num_items = 100
    num_contexts = 10
    vocabulary_sizes = {
        'num_categories': 20,
        'num_price_buckets': 5,
        'num_rating_levels': 4
    }
    
    # Create similarity matrix
    similarity_matrix = torch.rand(num_contexts, num_contexts)
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) / 2
    similarity_matrix.fill_diagonal_(1.0)
    
    # Create model
    model = CoFARSSparse(config, num_items, num_contexts, vocabulary_sizes, similarity_matrix)
    
    # Create test batch
    batch_size = 8
    max_seq_len = 10
    
    batch = {
        'user_segments': torch.tensor([0, 0, 1, 1, 1, 2, 2, 2]),
        'user_item_ids': torch.randint(0, num_items, (batch_size, max_seq_len)),
        'sequence_lengths': torch.tensor([7, 8, 3, 2, 4, 1, 1, 1]),
        'context_ids': torch.randint(0, num_contexts, (batch_size,)),
        'candidate_item_ids': torch.randint(0, num_items, (batch_size,)),
        'labels': torch.randint(0, 2, (batch_size,)).float()
    }
    
    # Forward pass
    scores, auxiliary = model(batch)
    print(f"Scores shape: {scores.shape}")
    print(f"Scores: {scores}")
    
    # Calculate loss
    total_loss, loss_dict = model.calculate_loss(batch, scores, auxiliary)
    print(f"\nLosses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # Test prediction
    probs = model.predict(batch)
    print(f"\nPrediction probabilities: {probs}")
    
    print("\n✅ CoFARS-Sparse model test passed!")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
