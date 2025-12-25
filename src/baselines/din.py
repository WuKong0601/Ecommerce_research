"""
DIN (Deep Interest Network) Baseline
Attention-based recommendation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention mechanism for DIN"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
    
    def forward(self, query, keys, sequence_lengths):
        """
        Args:
            query: [B, emb_dim] - candidate item
            keys: [B, seq_len, emb_dim] - history items
            sequence_lengths: [B]
        Returns:
            user_repr: [B, emb_dim]
        """
        batch_size, seq_len, emb_dim = keys.size()
        
        # Expand query to match keys
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Interaction features
        interaction = torch.cat([
            query_expanded,
            keys,
            query_expanded * keys,
            query_expanded - keys
        ], dim=-1)  # [B, seq_len, emb_dim * 4]
        
        # Attention scores
        attn_scores = self.attention(interaction).squeeze(-1)  # [B, seq_len]
        
        # Mask padding
        mask = torch.arange(seq_len, device=keys.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, seq_len]
        
        # Weighted sum
        user_repr = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)  # [B, emb_dim]
        return user_repr


class DINBaseline(nn.Module):
    """Deep Interest Network with attention mechanism"""
    def __init__(self, num_items, embedding_dim=16):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.attention = AttentionLayer(embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - user_item_ids: [B, seq_len]
                - candidate_item_ids: [B]
                - sequence_lengths: [B]
        Returns:
            scores: [B]
        """
        user_item_ids = batch['user_item_ids']
        candidate_item_ids = batch['candidate_item_ids']
        sequence_lengths = batch['sequence_lengths']
        
        # Get embeddings
        hist_emb = self.item_embedding(user_item_ids)  # [B, seq_len, emb_dim]
        cand_emb = self.item_embedding(candidate_item_ids)  # [B, emb_dim]
        
        # Attention-based user representation
        user_repr = self.attention(cand_emb, hist_emb, sequence_lengths)
        
        # Concatenate and predict
        combined = torch.cat([user_repr, cand_emb], dim=-1)
        score = self.fc(combined).squeeze(-1)
        return score
