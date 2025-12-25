"""
Unified Training Script for Baseline Models
Trains all 3 baselines and saves results for paper comparison
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.baselines import AveragePoolingBaseline, StandardGRUBaseline, DINBaseline
from src.data_processing.dataset import create_dataloaders
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward
            scores = model(batch)
            labels = batch['labels']
            loss = criterion(scores, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """Evaluate model"""
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward
            scores = model(batch)
            labels = batch['labels']
            loss = criterion(scores, labels)
            
            total_loss += loss.item()
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = 0.5
    
    try:
        ap = average_precision_score(all_labels, all_scores)
    except:
        ap =1.0
    
    return {
        'loss': total_loss / len(val_loader),
        'auc': auc,
        'ap': ap
    }


def train_baseline(model_name, model, train_loader, val_loader, test_loader, 
                   epochs, lr, device, logger):
    """
    Train a single baseline model
    
    Args:
        model_name: Name of the model
        model: Model instance
        train_loader, val_loader, test_loader: DataLoaders
        epochs: Number of epochs
        lr: Learning rate
        device: Device
        logger: Logger
        
    Returns:
        results: Dictionary with training results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_aucs': [],
        'val_aps': []
    }
    
    best_val_auc = 0
    models_dir = 'results/models/baselines'
    os.makedirs(models_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val AP: {val_metrics['ap']:.4f}")
        
        # Save history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_metrics['loss'])
        history['val_aucs'].append(val_metrics['auc'])
        history['val_aps'].append(val_metrics['ap'])
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_path = os.path.join(models_dir, f'{model_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc
            }, best_model_path)
            logger.info(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
    
    # Test with best model
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {model_name}")
    logger.info(f"{'='*80}")
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test AP: {test_metrics['ap']:.4f}")
    
    return {
        'model_name': model_name,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'training_history': history,
        'num_params': sum(p.numel() for p in model.parameters())
    }


def save_comparison_results(all_results, cofars_results):
    """Save comparison table for paper"""
    
    # Create comparison table
    comparison = {
        'baselines': [],
        'cofars_sparse': cofars_results
    }
    
    for result in all_results:
        comparison['baselines'].append({
            'name': result['model_name'],
            'num_params': result['num_params'],
            'best_val_auc': result['best_val_auc'],
            'test_auc': result['test_metrics']['auc'],
            'test_ap': result['test_metrics']['ap'],
            'test_loss': result['test_metrics']['loss']
        })
    
    # Save JSON
    save_path = 'results/statistics/baselines_comparison.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create markdown table
    md_table = "# Baseline Comparison Results\n\n"
    md_table += "| Model | Parameters | Test AUC | Test AP | vs CoFARS-Sparse |\n"
    md_table += "|-------|------------|----------|---------|------------------|\n"
    
    for result in comparison['baselines']:
        improvement = ((cofars_results['test_auc'] - result['test_auc']) / result['test_auc']) * 100
        md_table += f"| {result['name']} | {result['num_params']:,} | {result['test_auc']:.4f} | {result['test_ap']:.4f} | +{improvement:.1f}% |\n"
    
    md_table += f"| **CoFARS-Sparse** | **{cofars_results['num_params']:,}** | **{cofars_results['test_auc']:.4f}** | **{cofars_results['test_ap']:.4f}** | **Baseline** |\n"
    
    # Save markdown
    md_path = 'results/statistics/baselines_comparison.md'
    with open(md_path, 'w') as f:
        f.write(md_table)
    
    print(f"\n✓ Saved comparison results to:")
    print(f"  - {save_path}")
    print(f"  - {md_path}")


def main():
    # Load config
    config = load_config()
    
    # Setup logger
    logger = setup_logger('baselines_training')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data (same as CoFARS-Sparse)
    logger.info("Loading data...")
    data_dir = config['data']['processed_dir']
    batch_size = config['training']['batch_size']
    neg_sample_ratio = config['training'].get('neg_sample_ratio', 4)
    
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir, 
        batch_size=batch_size,
        neg_sample_ratio=neg_sample_ratio
    )
    
    logger.info(f"Data loaded:")
    logger.info(f"  Train: {metadata['train_size']} samples")
    logger.info(f"  Val: {metadata['val_size']} samples")
    logger.info(f"  Test: {metadata['test_size']} samples")
    logger.info(f"  Items: {metadata['num_items']}")
    
    # Models to train
    models = {
        'average_pooling': AveragePoolingBaseline(metadata['num_items']),
        'standard_gru': StandardGRUBaseline(metadata['num_items']),
        'din': DINBaseline(metadata['num_items'])
    }
    
    # Training config
    epochs = 30  # Fewer than CoFARS-Sparse
    lr = 5e-4
    
    # Train all baselines
    all_results = []
    for model_name, model in models.items():
        result = train_baseline(
            model_name, model, train_loader, val_loader, test_loader,
            epochs, lr, device, logger
        )
        all_results.append(result)
    
    # Load CoFARS-Sparse results for comparison
    cofars_results_path = 'results/statistics/training_results.json'
    with open(cofars_results_path, 'r') as f:
        cofars_data = json.load(f)
    
    cofars_results = {
        'name': 'CoFARS-Sparse',
        'num_params': cofars_data['model_params'],
        'best_val_auc': cofars_data['best_val_auc'],
        'test_auc': cofars_data['test_metrics']['auc'],
        'test_ap': cofars_data['test_metrics']['ap'],
        'test_loss': cofars_data['test_metrics']['loss']
    }
    
    # Save comparison
    save_comparison_results(all_results, cofars_results)
    
    logger.info("\n" + "="*80)
    logger.info("ALL BASELINE TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
