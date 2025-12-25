"""
Dataset and DataLoader for CoFARS-Sparse
Handles data loading for hybrid user modeling
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CoFARSDataset(Dataset):
    """
    Dataset for CoFARS-Sparse training
    """
    
    def __init__(self, reviews_df, user_segments_df, products_df, 
                 context_mapping, max_seq_len=50, mode='train', neg_sample_ratio=4):
        """
        Args:
            reviews_df: Reviews with contexts
            user_segments_df: User segmentation info
            products_df: Product information
            context_mapping: Context to ID mapping
            max_seq_len: Maximum sequence length
            mode: 'train', 'val', or 'test'
            neg_sample_ratio: Number of negative samples per positive sample
        """
        self.reviews = reviews_df
        self.user_segments = user_segments_df
        self.products = products_df
        self.context_mapping = context_mapping
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.neg_sample_ratio = neg_sample_ratio
        
        # Create product ID mapping (raw ID -> 0-indexed)
        unique_product_ids = sorted(products_df['id'].unique())
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(unique_product_ids)}
        self.idx_to_product_id = {idx: pid for pid, idx in self.product_id_to_idx.items()}
        self.num_items = len(unique_product_ids)
        self.all_item_ids = list(range(self.num_items))  # For negative sampling
        
        # Create user segment mappings
        self.user_to_segment = dict(zip(user_segments_df['customer_id'], 
                                       user_segments_df['segment']))
        self.segment_to_id = {'power': 0, 'regular': 1, 'cold_start': 2}
        
        # Map contexts to IDs
        self.context_to_id = {ctx: i for i, ctx in enumerate(sorted(set(reviews_df['context'])))}
        self.id_to_context = {i: ctx for ctx, i in self.context_to_id.items()}
        
        # Prepare user sequences
        self.user_sequences = self._prepare_sequences()
        
    def _prepare_sequences(self):
        """Prepare user interaction sequences with negative sampling"""
        import random
        sequences = []
        
        # Sort by timestamp
        reviews_sorted = self.reviews.sort_values('created_at_raw')
        
        for user_id, group in reviews_sorted.groupby('customer_id'):
            # Convert raw product IDs to 0-indexed
            items = [self.product_id_to_idx[pid] for pid in group['product_id'].tolist()]
            contexts = group['context'].tolist()
            ratings = group['rating'].tolist()
            
            # Get user segment
            segment = self.user_to_segment.get(user_id, 'cold_start')
            segment_id = self.segment_to_id[segment]
            
            # Items the user has interacted with (for negative sampling exclusion)
            user_interacted_items = set(items)
            
            # For each interaction, create positive and negative samples
            for i in range(len(items)):
                # Historical items (up to current)
                hist_items = items[:i+1]
                hist_len = len(hist_items)
                
                # Current context
                current_context = contexts[i]
                context_id = self.context_to_id[current_context]
                
                # Positive sample
                target_item = items[i]
                sequences.append({
                    'user_id': user_id,
                    'segment_id': segment_id,
                    'hist_items': hist_items,
                    'hist_len': hist_len,
                    'context_id': context_id,
                    'target_item': target_item,
                    'label': 1
                })
                
                # Negative samples (for training, val, and test to enable AUC calculation)
                for _ in range(self.neg_sample_ratio):
                    # Sample a random item the user hasn't interacted with
                    neg_item = random.choice(self.all_item_ids)
                    while neg_item in user_interacted_items:
                        neg_item = random.choice(self.all_item_ids)
                    
                    sequences.append({
                        'user_id': user_id,
                        'segment_id': segment_id,
                        'hist_items': hist_items,
                        'hist_len': hist_len,
                        'context_id': context_id,
                        'target_item': neg_item,
                        'label': 0  # Negative sample
                    })
        
        return sequences
    
    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, idx):
        sample = self.user_sequences[idx]
        
        # Pad/truncate historical items
        hist_items = sample['hist_items'][-self.max_seq_len:]  # Take last N
        hist_len = len(hist_items)
        
        # Pad if needed
        if hist_len < self.max_seq_len:
            hist_items = [0] * (self.max_seq_len - hist_len) + hist_items
        
        return {
            'user_id': sample['user_id'],
            'segment_id': sample['segment_id'],
            'hist_items': torch.LongTensor(hist_items),
            'hist_len': hist_len,
            'context_id': sample['context_id'],
            'target_item': sample['target_item'],
            'label': sample['label']
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    return {
        'user_ids': [item['user_id'] for item in batch],
        'user_segments': torch.LongTensor([item['segment_id'] for item in batch]),
        'user_item_ids': torch.stack([item['hist_items'] for item in batch]),
        'sequence_lengths': torch.LongTensor([item['hist_len'] for item in batch]),
        'context_ids': torch.LongTensor([item['context_id'] for item in batch]),
        'candidate_item_ids': torch.LongTensor([item['target_item'] for item in batch]),
        'labels': torch.FloatTensor([item['label'] for item in batch])
    }


def create_dataloaders(data_dir, batch_size=128, num_workers=0, neg_sample_ratio=4):
    """
    Create train/val/test dataloaders
    
    Args:
        data_dir: Path to processed data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        neg_sample_ratio: Number of negative samples per positive (for training)
        
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    # Load data
    contexts_dir = os.path.join(data_dir, 'contexts')
    segmentation_dir = os.path.join(data_dir, 'segmentation')
    
    reviews = pd.read_csv(os.path.join(contexts_dir, 'reviews_with_contexts.csv'))
    products = pd.read_csv(os.path.join(contexts_dir, 'products_with_attributes.csv'))
    user_segments = pd.read_csv(os.path.join(segmentation_dir, 'user_segments.csv'))
    
    with open(os.path.join(contexts_dir, 'context_mapping.json'), 'r') as f:
        context_mapping = json.load(f)
    
    # Split by users (80/10/10)
    unique_users = reviews['customer_id'].unique()
    train_users, temp_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)
    
    # Split reviews
    train_reviews = reviews[reviews['customer_id'].isin(train_users)]
    val_reviews = reviews[reviews['customer_id'].isin(val_users)]
    test_reviews = reviews[reviews['customer_id'].isin(test_users)]
    
    # Create datasets (all sets use negative samples for proper evaluation)
    train_dataset = CoFARSDataset(train_reviews, user_segments, products, context_mapping, 
                                  mode='train', neg_sample_ratio=neg_sample_ratio)
    val_dataset = CoFARSDataset(val_reviews, user_segments, products, context_mapping, 
                                mode='val', neg_sample_ratio=neg_sample_ratio)
    test_dataset = CoFARSDataset(test_reviews, user_segments, products, context_mapping, 
                                 mode='test', neg_sample_ratio=neg_sample_ratio)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    
    # Metadata
    metadata = {
        'num_users': len(unique_users),
        'num_items': len(products),  # Number of unique products
        'num_contexts': len(train_dataset.context_to_id),
        'context_to_id': train_dataset.context_to_id,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    print("Testing CoFARS Dataset...")
    
    # Test with actual data
    data_dir = "../../processed_data"
    
    try:
        train_loader, val_loader, test_loader, metadata = create_dataloaders(
            data_dir, batch_size=4
        )
        
        print(f"Metadata:")
        for k, v in metadata.items():
            if k != 'context_to_id':
                print(f"  {k}: {v}")
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch size: {batch['user_segments'].shape[0]}")
        print(f"Segments in batch: {batch['user_segments']}")
        print(f"Sequence lengths: {batch['sequence_lengths']}")
        
        print("\n✅ Dataset test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: Run from project root or ensure data paths are correct")
