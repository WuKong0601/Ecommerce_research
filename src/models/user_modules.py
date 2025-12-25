"""
Hybrid User Modeling Modules for CoFARS-Sparse
Different strategies for power, regular, and cold-start users
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerUserModule(nn.Module):
    """
    Full sequential modeling for power users (≥5 interactions)
    Uses GRU to model sequence evolution
    """
    
    def __init__(self, item_embedding_dim, hidden_dim, num_layers=1):
        super(PowerUserModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=item_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention for aggregating sequence
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, item_sequence_embeddings, sequence_lengths=None):
        """
        Args:
            item_sequence_embeddings: (batch_size, max_seq_len, item_emb_dim)
            sequence_lengths: (batch_size,) actual sequence lengths
            
        Returns:
            user_representation: (batch_size, hidden_dim)
        """
        # GRU encoding
        gru_output, _ = self.gru(item_sequence_embeddings)  # (batch_size, seq_len, hidden_dim)
        
        # Attention-based aggregation
        attention_scores = self.attention(gru_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        user_representation = (gru_output * attention_weights).sum(dim=1)  # (batch_size, hidden_dim)
        
        return user_representation


class RegularUserModule(nn.Module):
    """
    Hybrid approach for regular users (2-4 interactions)
    Combines simple user preference with context information
    """
    
    def __init__(self, item_embedding_dim, context_embedding_dim, output_dim):
        super(RegularUserModule, self).__init__()
        
        # Simple aggregation of user's items
        self.user_encoder = nn.Sequential(
            nn.Linear(item_embedding_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_embedding_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, item_embeddings, context_embedding):
        """
        Args:
            item_embeddings: (batch_size, num_items, item_emb_dim)
            context_embedding: (batch_size, context_emb_dim)
            
        Returns:
            user_representation: (batch_size, output_dim)
        """
        # Average user's items
        user_pref = item_embeddings.mean(dim=1)  # (batch_size, item_emb_dim)
        user_encoded = self.user_encoder(user_pref)  # (batch_size, output_dim)
        
        # Encode context
        context_encoded = self.context_encoder(context_embedding)  # (batch_size, output_dim)
        
        # Combine
        combined = torch.cat([user_encoded, context_encoded], dim=-1)  # (batch_size, output_dim*2)
        user_representation = self.fusion(combined)  # (batch_size, output_dim)
        
        return user_representation


class ColdStartModule(nn.Module):
    """
    Pure context-based prediction for cold-start users (1 interaction)
    """
    
    def __init__(self, context_embedding_dim, output_dim):
        super(ColdStartModule, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(context_embedding_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, context_embedding):
        """
        Args:
            context_embedding: (batch_size, context_emb_dim)
            
        Returns:
            user_representation: (batch_size, output_dim)
        """
        return self.encoder(context_embedding)


class HybridUserEncoder(nn.Module):
    """
    Routes users to appropriate module based on their segment
    Combines power, regular, and cold-start modules
    """
    
    def __init__(self, item_embedding_dim, context_embedding_dim, output_dim):
        super(HybridUserEncoder, self).__init__()
        
        self.item_embedding_dim = item_embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.output_dim = output_dim
        
        # Three modules
        self.power_module = PowerUserModule(item_embedding_dim, output_dim)
        self.regular_module = RegularUserModule(item_embedding_dim, context_embedding_dim, output_dim)
        self.coldstart_module = ColdStartModule(context_embedding_dim, output_dim)
        
    def forward(self, user_segment, item_embeddings, context_embedding, sequence_lengths=None):
        """
        Args:
            user_segment: (batch_size,) segment indices (0=power, 1=regular, 2=cold_start)
            item_embeddings: (batch_size, max_seq_len, item_emb_dim)
            context_embedding: (batch_size, context_emb_dim)
            sequence_lengths: (batch_size,) actual lengths
            
        Returns:
            user_representations: (batch_size, output_dim)
        """
        batch_size = user_segment.shape[0]
        device = user_segment.device
        
        user_representations = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Process each segment
        power_mask = (user_segment == 0)
        regular_mask = (user_segment == 1)
        coldstart_mask = (user_segment == 2)
        
        # Power users
        if power_mask.any():
            power_items = item_embeddings[power_mask]
            power_lens = sequence_lengths[power_mask] if sequence_lengths is not None else None
            power_repr = self.power_module(power_items, power_lens)
            user_representations[power_mask] = power_repr
        
        # Regular users
        if regular_mask.any():
            regular_items = item_embeddings[regular_mask]
            regular_context = context_embedding[regular_mask]
            regular_repr = self.regular_module(regular_items, regular_context)
            user_representations[regular_mask] = regular_repr
        
        # Cold-start users
        if coldstart_mask.any():
            coldstart_context = context_embedding[coldstart_mask]
            coldstart_repr = self.coldstart_module(coldstart_context)
            user_representations[coldstart_mask] = coldstart_repr
        
        return user_representations


if __name__ == "__main__":
    print("Testing Hybrid User Modules...")
    
    batch_size = 8
    max_seq_len = 10
    item_emb_dim = 32
    context_emb_dim = 64
    output_dim = 64
    
    # Create hybrid encoder
    encoder = HybridUserEncoder(item_emb_dim, context_emb_dim, output_dim)
    
    # Test data
    user_segments = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])  # Mix of segments
    item_embs = torch.randn(batch_size, max_seq_len, item_emb_dim)
    context_embs = torch.randn(batch_size, context_emb_dim)
    seq_lengths = torch.tensor([7, 8, 3, 2, 4, 1, 1, 1])
    
    # Forward pass
    user_reprs = encoder(user_segments, item_embs, context_embs, seq_lengths)
    
    print(f"User representations shape: {user_reprs.shape}")
    print(f"Power users (indices 0,1): {user_reprs[:2].shape}")
    print(f"Regular users (indices 2-4): {user_reprs[2:5].shape}")
    print(f"Cold-start users (indices 5-7): {user_reprs[5:].shape}")
    
    # Test individual modules
    print("\nTesting individual modules...")
    
    # Power
    power_mod = PowerUserModule(item_emb_dim, output_dim)
    power_out = power_mod(item_embs[:2])
    print(f"Power module output: {power_out.shape}")
    
    # Regular
    regular_mod = RegularUserModule(item_emb_dim, context_emb_dim, output_dim)
    regular_out = regular_mod(item_embs[:3], context_embs[:3])
    print(f"Regular module output: {regular_out.shape}")
    
    # Cold-start
    cold_mod = ColdStartModule(context_emb_dim, output_dim)
    cold_out = cold_mod(context_embs[:2])
    print(f"Cold-start module output: {cold_out.shape}")
    
    print("\n✅ Hybrid User Modules test passed!")
