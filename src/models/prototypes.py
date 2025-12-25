"""
Context-Aggregated Prototypes for CoFARS-Sparse
Learns prototypes from aggregated context preferences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextPrototypes(nn.Module):
    """
    Learnable prototypes that represent clusters of similar contexts
    
    Unlike original CoFARS where prototypes cluster user-context visits,
    here prototypes cluster context-level aggregated preferences
    """
    
    def __init__(self, num_prototypes, embedding_dim, num_contexts):
        """
        Args:
            num_prototypes: Number of prototype vectors
            embedding_dim: Dimension of embeddings
            num_contexts: Number of distinct contexts
        """
        super(ContextPrototypes, self).__init__()
        
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim
        self.num_contexts = num_contexts
        
        # Learnable prototype embeddings
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Context embeddings (will be updated during training)
        self.context_embeddings = nn.Embedding(num_contexts, embedding_dim)
        
    def forward(self, context_ids):
        """
        Get context embeddings
        
        Args:
            context_ids: (batch_size,) context indices
            
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.context_embeddings(context_ids)
    
    def get_prototype_assignment(self, context_embedding, temperature=1.0):
        """
        Soft assignment of context to prototypes using attention
        
        Args:
            context_embedding: (batch_size, embedding_dim)
            temperature: Temperature for softmax
            
        Returns:
            assignment: (batch_size, num_prototypes) soft assignment weights
        """
        # Calculate similarity to each prototype
        # (batch_size, embedding_dim) x (embedding_dim, num_prototypes)
        similarities = torch.matmul(context_embedding, self.prototypes.t())
        
        # Apply temperature and softmax
        assignment = F.softmax(similarities / temperature, dim=-1)
        
        return assignment
    
    def get_prototype_context_representation(self, context_embedding):
        """
        Get context representation weighted by prototype assignment
        
        Args:
            context_embedding: (batch_size, embedding_dim)
            
        Returns:
            prototype_weighted: (batch_size, embedding_dim)
        """
        # Get soft assignment
        assignment = self.get_prototype_assignment(context_embedding)
        
        # Weighted sum of prototypes
        # (batch_size, num_prototypes) x (num_prototypes, embedding_dim)
        prototype_weighted = torch.matmul(assignment, self.prototypes)
        
        return prototype_weighted


class IndependenceLoss(nn.Module):
    """
    Encourages prototypes to be different from each other
    From paper Equation 6: L_IND
    """
    
    def __init__(self):
        super(IndependenceLoss, self).__init__()
        
    def forward(self, prototypes):
        """
        Args:
            prototypes: (num_prototypes, embedding_dim)
            
        Returns:
            loss: Independence loss (lower = more different prototypes)
        """
        # Normalize prototypes
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # Calculate pairwise similarities
        similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.t())
        
        # Remove diagonal (self-similarity)
        mask = 1 - torch.eye(prototypes.shape[0], device=prototypes.device)
        
        # Independence loss: minimize similarity between different prototypes
        loss = (similarity_matrix * mask).pow(2).sum() / (prototypes.shape[0] * (prototypes.shape[0] - 1))
        
        return loss


class StaticContextMatcher(nn.Module):
    """
    Uses pre-computed JS divergence similarity matrix for context matching
    Simplified version replacing temporal graph from original paper
    """
    
    def __init__(self, similarity_matrix):
        """
        Args:
            similarity_matrix: (num_contexts, num_contexts) pre-computed similarity
        """
        super(StaticContextMatcher, self).__init__()
        
        # Register as buffer (not trained)
        self.register_buffer('similarity_matrix', similarity_matrix)
        
    def get_similar_contexts(self, context_id, top_k=3):
        """
        Get top-k similar contexts
        
        Args:
            context_id: (batch_size,) or int
            top_k: Number of similar contexts to return
            
        Returns:
            similar_contexts: (batch_size, top_k) indices
            similarities: (batch_size, top_k) similarity scores
        """
        if isinstance(context_id, int):
            context_id = torch.tensor([context_id], device=self.similarity_matrix.device)
        
        # Get similarities for given contexts
        sims = self.similarity_matrix[context_id]  # (batch_size, num_contexts)
        
        # Get top-k (excluding self at index 0)
        top_k_sims, top_k_indices = torch.topk(sims, k=top_k+1, dim=-1)
        
        # Remove self (highest similarity)
        similar_contexts = top_k_indices[:, 1:]
        similarities = top_k_sims[:, 1:]
        
        return similar_contexts, similarities
    
    def aggregate_from_similar_contexts(self, context_embedding, context_id, 
                                       context_prototypes, top_k=3):
        """
        Aggregate information from similar contexts
        
        Args:
            context_embedding: (batch_size, embedding_dim) current context
            context_id: (batch_size,) context indices
            context_prototypes: ContextPrototypes instance
            top_k: Number of similar contexts
            
        Returns:
            aggregated: (batch_size, embedding_dim) enriched representation
        """
        # Get similar contexts
        similar_ctx_ids, similarities = self.get_similar_contexts(context_id, top_k)
        
        # Get embeddings of similar contexts
        batch_size = context_embedding.shape[0]
        similar_embeddings = []
        
        for i in range(batch_size):
            sim_embs = context_prototypes(similar_ctx_ids[i])  # (top_k, embedding_dim)
            similar_embeddings.append(sim_embs)
        
        similar_embeddings = torch.stack(similar_embeddings)  # (batch_size, top_k, embedding_dim)
        
        # Weighted aggregation
        similarities = similarities.unsqueeze(-1)  # (batch_size, top_k, 1)
        weighted_similar = (similar_embeddings * similarities).sum(dim=1)  # (batch_size, embedding_dim)
        
        # Combine with current context
        aggregated = context_embedding + 0.3 * weighted_similar
        
        return aggregated


if __name__ == "__main__":
    print("Testing Context Prototypes...")
    
    num_prototypes = 30
    embedding_dim = 64
    num_contexts = 10
    batch_size = 4
    
    # Create prototypes
    prototypes = ContextPrototypes(num_prototypes, embedding_dim, num_contexts)
    
    # Test forward
    context_ids = torch.randint(0, num_contexts, (batch_size,))
    context_emb = prototypes(context_ids)
    print(f"Context embeddings shape: {context_emb.shape}")
    
    # Test prototype assignment
    assignment = prototypes.get_prototype_assignment(context_emb)
    print(f"Prototype assignment shape: {assignment.shape}")
    print(f"Assignment sums to 1: {assignment.sum(dim=-1)}")
    
    # Test independence loss
    ind_loss = IndependenceLoss()
    loss = ind_loss(prototypes.prototypes)
    print(f"Independence loss: {loss.item():.6f}")
    
    # Test static matcher
    similarity_matrix = torch.rand(num_contexts, num_contexts)
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) / 2  # Symmetric
    similarity_matrix.fill_diagonal_(1.0)  # Self-similarity = 1
    
    matcher = StaticContextMatcher(similarity_matrix)
    similar_ctx, sims = matcher.get_similar_contexts(context_ids[0].item(), top_k=3)
    print(f"\nSimilar contexts for context {context_ids[0].item()}: {similar_ctx}")
    print(f"Similarities: {sims}")
    
    # Test aggregation
    aggregated = matcher.aggregate_from_similar_contexts(
        context_emb, context_ids, prototypes, top_k=3
    )
    print(f"Aggregated shape: {aggregated.shape}")
    
    print("\n✅ Context Prototypes test passed!")
