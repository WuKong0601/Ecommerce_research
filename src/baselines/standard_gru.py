"""
Standard GRU Baseline
GRU-based sequence modeling without context information
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StandardGRUBaseline(nn.Module):
    def __init__(self, num_items, embedding_dim=16, hidden_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            batch_first=True,
            num_layers=1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 64),
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
        
        # Pack padded sequence
        packed_input = pack_padded_sequence(
            hist_emb, 
            sequence_lengths.cpu().clamp(min=1), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # GRU encoding
        _, hidden = self.gru(packed_input)
        user_repr = hidden.squeeze(0)  # [B, hidden_dim]
        
        # Concatenate and predict
        combined = torch.cat([user_repr, cand_emb], dim=-1)
        score = self.fc(combined).squeeze(-1)
        return score
