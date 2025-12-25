"""
Probability Encoder for CoFARS-Sparse
Maps context embeddings to probability distributions over product attributes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilityEncoder(nn.Module):
    """
    Encodes context representations into probability distributions
    over product attributes (category, price, rating)
    
    From paper Section 3.2:
    - Maps latent representations to probability distributions
    - Aligns estimated JS divergence with ground truth using MSE loss
    """
    
    def __init__(self, input_dim, num_categories, num_price_buckets, num_rating_levels, hidden_dim=128):
        """
        Args:
            input_dim: Dimension of context embedding
            num_categories: Number of product categories
            num_price_buckets: Number of price buckets
            num_rating_levels: Number of rating levels
            hidden_dim: Hidden layer dimension
        """
        super(ProbabilityEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_price_buckets = num_price_buckets
        self.num_rating_levels = num_rating_levels
        
        # Total output dimension
        self.output_dim = num_categories + num_price_buckets + num_rating_levels
        
        # MLP layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
    def forward(self, context_embedding):
        """
        Args:
            context_embedding: (batch_size, input_dim)
            
        Returns:
            distributions: Dict with category, price, rating distributions
        """
        # Encode to logits
        logits = self.encoder(context_embedding)
        
        # Split into separate distributions
        category_logits = logits[:, :self.num_categories]
        price_logits = logits[:, self.num_categories:self.num_categories + self.num_price_buckets]
        rating_logits = logits[:, self.num_categories + self.num_price_buckets:]
        
        # Apply softmax to get probability distributions
        category_dist = F.softmax(category_logits, dim=-1)
        price_dist = F.softmax(price_logits, dim=-1)
        rating_dist = F.softmax(rating_logits, dim=-1)
        
        return {
            'category': category_dist,
            'price': price_dist,
            'rating': rating_dist,
            'combined': torch.cat([category_dist, price_dist, rating_dist], dim=-1)
        }


def calculate_kl_divergence(P, Q, epsilon=1e-8):
    """
    Calculate KL divergence KL(P||Q)
    
    Args:
        P, Q: Probability distributions (batch_size, dim)
        epsilon: Small constant to avoid log(0)
        
    Returns:
        kl: KL divergence values (batch_size,)
    """
    P = P + epsilon
    Q = Q + epsilon
    
    # Normalize
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    
    # KL(P||Q) = sum(P * log(P/Q))
    kl = (P * torch.log(P / Q)).sum(dim=-1)
    
    return kl


def calculate_js_divergence(P, Q, epsilon=1e-8):
    """
    Calculate Jensen-Shannon divergence
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Args:
        P, Q: Probability distributions (batch_size, dim)
        epsilon: Small constant
        
    Returns:
        js: JS divergence values (batch_size,)
    """
    # Calculate midpoint
    M = 0.5 * (P + Q)
    
    # Calculate JS divergence
    js = 0.5 * calculate_kl_divergence(P, M, epsilon) + \
         0.5 * calculate_kl_divergence(Q, M, epsilon)
    
    return js


class JSDivergenceLoss(nn.Module):
    """
    MSE loss between estimated JS divergence and ground truth
    From paper Equation 5
    """
    
    def __init__(self):
        super(JSDivergenceLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, context_emb_i, context_emb_j, prob_encoder, ground_truth_js):
        """
        Args:
            context_emb_i: Context i embeddings (batch_size, dim)
            context_emb_j: Context j embeddings (batch_size, dim)
            prob_encoder: ProbabilityEncoder instance
            ground_truth_js: Ground truth JS divergence (batch_size,)
            
        Returns:
            loss: MSE loss
        """
        # Get probability distributions
        dist_i = prob_encoder(context_emb_i)
        dist_j = prob_encoder(context_emb_j)
        
        # Calculate estimated JS divergence
        estimated_js = calculate_js_divergence(dist_i['combined'], dist_j['combined'])
        
        # MSE loss
        loss = self.mse(estimated_js, ground_truth_js)
        
        return loss, estimated_js


if __name__ == "__main__":
    # Test the probability encoder
    print("Testing Probability Encoder...")
    
    batch_size = 4
    input_dim = 64
    num_categories = 20
    num_price_buckets = 5
    num_rating_levels = 4
    
    # Create encoder
    encoder = ProbabilityEncoder(
        input_dim=input_dim,
        num_categories=num_categories,
        num_price_buckets=num_price_buckets,
        num_rating_levels=num_rating_levels
    )
    
    # Test forward pass
    context_emb = torch.randn(batch_size, input_dim)
    distributions = encoder(context_emb)
    
    print(f"Category dist shape: {distributions['category'].shape}")
    print(f"Price dist shape: {distributions['price'].shape}")
    print(f"Rating dist shape: {distributions['rating'].shape}")
    print(f"Combined dist shape: {distributions['combined'].shape}")
    
    # Test JS divergence
    context_emb_2 = torch.randn(batch_size, input_dim)
    dist_2 = encoder(context_emb_2)
    
    js = calculate_js_divergence(distributions['combined'], dist_2['combined'])
    print(f"\nJS divergence: {js}")
    
    # Test loss
    js_loss = JSDivergenceLoss()
    ground_truth = torch.tensor([0.001, 0.002, 0.0015, 0.0008])
    loss, estimated = js_loss(context_emb, context_emb_2, encoder, ground_truth)
    print(f"Loss: {loss.item():.6f}")
    print(f"Estimated JS: {estimated}")
    
    print("\n✅ Probability Encoder test passed!")
