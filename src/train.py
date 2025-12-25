"""
Training script for CoFARS-Sparse
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.models.cofars_sparse import CoFARSSparse
from src.data_processing.dataset import create_dataloaders

def train_epoch(model, train_loader, optimizer, device, config, js_matrix=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'rec': 0, 'js': 0, 'ind': 0}
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward
        scores, auxiliary = model(batch)
        loss, loss_dict = model.calculate_loss(batch, scores, auxiliary, js_matrix)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict[k]
        
        # Update progress
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def evaluate(model, val_loader, device, js_matrix=None):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            scores, auxiliary = model(batch)
            loss, _ = model.calculate_loss(batch, scores, auxiliary, js_matrix)
            
            total_loss += loss.item()
            
            # Predictions
            probs = torch.sigmoid(scores)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Handle case where all labels are the same
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5  # Default when AUC can't be calculated
    
    try:
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        ap = all_labels.mean()  # Default to label prevalence
    
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'ap': ap
    }
    
    return metrics


def main():
    """Main training loop"""
    # Load config
    config = load_config()
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("TRAINING COFARS-SPARSE")
    logger.info("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Loading data...")
    data_dir = config['data']['processed_dir']
    batch_size = config['training']['batch_size']
    neg_sample_ratio = config['training'].get('neg_sample_ratio', 4)
    
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir, batch_size=batch_size, neg_sample_ratio=neg_sample_ratio
    )
    
    logger.info(f"Data loaded:")
    logger.info(f"  Train: {metadata['train_size']} samples")
    logger.info(f"  Val: {metadata['val_size']} samples")
    logger.info(f"  Test: {metadata['test_size']} samples")
    logger.info(f"  Users: {metadata['num_users']}")
    logger.info(f"  Items: {metadata['num_items']}")
    logger.info(f"  Contexts: {metadata['num_contexts']}")
    
    # Load JS similarity matrix
    divergence_dir = os.path.join(data_dir, 'divergences')
    js_matrix_np = np.load(os.path.join(divergence_dir, 'js_divergence_matrix.npy'))
    js_matrix = torch.FloatTensor(js_matrix_np).to(device)
    
    # Load vocabulary sizes
    context_agg_dir = os.path.join(data_dir, 'context_aggregation')
    with open(os.path.join(context_agg_dir, 'vocabulary.json'), 'r') as f:
        vocabulary = json.load(f)
    
    vocabulary_sizes = {
        'num_categories': len(vocabulary['categories']),
        'num_price_buckets': len(vocabulary['price_buckets']),
        'num_rating_levels': len(vocabulary['rating_levels'])
    }
    
    # Create similarity matrix
    similarity_matrix = 1 - js_matrix
    
    # Create model
    logger.info("Creating model...")
    model = CoFARSSparse(
        config=config,
        num_items=metadata['num_items'],
        num_contexts=metadata['num_contexts'],
        vocabulary_sizes=vocabulary_sizes,
        similarity_matrix=similarity_matrix
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 1e-5))
    )
    
    # Learning rate scheduler
    scheduler = None
    if 'lr_scheduler' in config['training']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize AUC
            patience=int(config['training']['lr_scheduler']['patience']),
            factor=float(config['training']['lr_scheduler']['factor']),
            min_lr=float(config['training']['lr_scheduler']['min_lr']),
            verbose=True
        )
        logger.info("Learning rate scheduler enabled (ReduceLROnPlateau)")
    
    # Early stopping
    early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 10)
    early_stopping_min_delta = config['training'].get('early_stopping', {}).get('min_delta', 0.001)
    early_stopping_counter = 0
    logger.info(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    # Training loop
    epochs = config['training']['epochs']
    best_val_auc = 0
    models_dir = config['results']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device, config, js_matrix
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"  Components: {train_components}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, js_matrix)
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val AP: {val_metrics['ap']:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['auc'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Save best model and check early stopping
        if val_metrics['auc'] > best_val_auc + early_stopping_min_delta:
            best_val_auc = val_metrics['auc']
            early_stopping_counter = 0
            
            best_model_path = os.path.join(models_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_auc': best_val_auc,
                'config': config
            }, best_model_path)
            logger.info(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        else:
            early_stopping_counter += 1
            logger.info(f"No improvement for {early_stopping_counter} epoch(s)")
        
        # Check early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            logger.info(f"Best validation AUC: {best_val_auc:.4f}")
            break
        
        # Save periodic checkpoints
        save_freq = config['training'].get('save_every_n_epochs', 5)
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_auc': val_metrics['auc'],
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best Val AUC: {best_val_auc:.4f}")
    logger.info("="*80)
    
    # Test
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, js_matrix)
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test AP: {test_metrics['ap']:.4f}")
    
    # Save results
    results = {
        'best_val_auc': best_val_auc,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'model_params': total_params,
        'config': config
    }
    
    results_path = os.path.join(config['results']['statistics_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
