"""
Average Pooling Baseline
Simplest baseline: Average all item embeddings in user history
"""

import torch
import torch.nn as nn


class AveragePoolingBaseline(nn.Module):
    def __init__(self, num_items, embedding_dim=16):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
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
        
        # Average pooling (handle variable lengths)
        batch_size, seq_len, emb_dim = hist_emb.size()
        mask = torch.arange(seq_len, device=hist_emb.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        
        hist_emb_masked = hist_emb * mask
        user_repr = hist_emb_masked.sum(dim=1) / (sequence_lengths.unsqueeze(1).float() + 1e-8)
        
        # Concatenate and predict
        combined = torch.cat([user_repr, cand_emb], dim=-1)
        score = self.fc(combined).squeeze(-1)
        return score
