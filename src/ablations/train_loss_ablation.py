"""
Loss Function Ablation Study for CoFARS-Sparse

Evaluates impact of each loss component:
- Full model: L_REC + γ*L_MSE + λ*L_IND
- w/o L_MSE: γ=0
- w/o L_IND: λ=0
- Only L_REC: γ=0, λ=0

Usage:
    python src/ablations/train_loss_ablation.py
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        scores, auxiliary = model(batch)
        loss, loss_dict = model.calculate_loss(batch, scores, auxiliary, js_matrix)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict[k]
    
    n_batches = len(train_loader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


def evaluate(model, val_loader, device, js_matrix=None):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            scores, auxiliary = model(batch)
            loss, _ = model.calculate_loss(batch, scores, auxiliary, js_matrix)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(scores)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    try:
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        ap = all_labels.mean()
    
    return {'loss': avg_loss, 'auc': auc, 'ap': ap}


def train_ablation_variant(variant_name, config, train_loader, val_loader, test_loader, 
                          metadata, js_matrix, device, logger):
    """Train a single ablation variant"""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING VARIANT: {variant_name}")
    logger.info(f"  γ (gamma) = {config['training']['gamma']}")
    logger.info(f"  λ (lambda) = {config['training']['lambda']}")
    logger.info(f"{'='*80}")
    
    # Load vocabulary sizes
    context_agg_dir = os.path.join(config['data']['processed_dir'], 'context_aggregation')
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
    model = CoFARSSparse(
        config=config,
        num_items=metadata['num_items'],
        num_contexts=metadata['num_contexts'],
        vocabulary_sizes=vocabulary_sizes,
        similarity_matrix=similarity_matrix
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 1e-5))
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6
    )
    
    # Training loop (reduced epochs for ablation study)
    epochs = 20  # Reduced from 50 for faster experiments
    best_val_auc = 0
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device, config, js_matrix
        )
        
        val_metrics = evaluate(model, val_loader, device, js_matrix)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss={train_loss:.4f}, "
                   f"Val AUC={val_metrics['auc']:.4f}, "
                   f"Val AP={val_metrics['ap']:.4f}")
        
        scheduler.step(val_metrics['auc'])
        
        if val_metrics['auc'] > best_val_auc + 0.001:
            best_val_auc = val_metrics['auc']
            early_stopping_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, js_matrix)
    
    logger.info(f"\n{variant_name} RESULTS:")
    logger.info(f"  Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Test AP: {test_metrics['ap']:.4f}")
    
    # Save model checkpoint
    models_dir = os.path.join(config['results']['models_dir'], 'ablations')
    os.makedirs(models_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(models_dir, f'{variant_name}_model.pt')
    torch.save({
        'model_state_dict': best_model_state,
        'val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'config': config
    }, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    return {
        'gamma': config['training']['gamma'],
        'lambda': config['training']['lambda'],
        'best_val_auc': best_val_auc,
        'test_auc': test_metrics['auc'],
        'test_ap': test_metrics['ap'],
        'test_loss': test_metrics['loss']
    }


def main():
    """Main ablation study"""
    # Load base config
    config = load_config()
    logger = setup_logger(config['logging']['log_dir'], config['logging']['log_level'])
    
    logger.info("="*80)
    logger.info("LOSS FUNCTION ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data (once for all experiments)
    logger.info("Loading data...")
    data_dir = config['data']['processed_dir']
    batch_size = config['training']['batch_size']
    neg_sample_ratio = config['training'].get('neg_sample_ratio', 4)
    
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir, batch_size=batch_size, neg_sample_ratio=neg_sample_ratio
    )
    
    logger.info(f"Data loaded: Train={metadata['train_size']}, Val={metadata['val_size']}, Test={metadata['test_size']}")
    
    # Load JS divergence matrix
    divergence_dir = os.path.join(data_dir, 'divergences')
    js_matrix_np = np.load(os.path.join(divergence_dir, 'js_divergence_matrix.npy'))
    js_matrix = torch.FloatTensor(js_matrix_np).to(device)
    
    # Define ablation variants
    ablation_variants = [
        ('without_js_loss', 0.0, 0.001),        # γ=0, λ=0.001: Remove L_MSE
        ('without_ind_loss', 0.05, 0.0),        # γ=0.05, λ=0: Remove L_IND
        ('only_rec_loss', 0.0, 0.0),            # γ=0, λ=0: Only L_REC
    ]
    
    # Results dictionary (include existing full model results)
    results = {
        'full_model': {
            'gamma': 0.05,
            'lambda': 0.001,
            'best_val_auc': 0.9287891628303351,
            'test_auc': 0.9329734277384424,
            'test_ap': 0.7557777761423418,
            'test_loss': 0.25495845879921714
        }
    }
    
    # Run each ablation variant
    for variant_name, gamma, lambda_val in ablation_variants:
        # Create config copy with modified loss weights
        variant_config = copy.deepcopy(config)
        variant_config['training']['gamma'] = gamma
        variant_config['training']['lambda'] = lambda_val
        
        results[variant_name] = train_ablation_variant(
            variant_name, variant_config, train_loader, val_loader, test_loader,
            metadata, js_matrix, device, logger
        )
    
    # Save results
    results_path = os.path.join(config['results']['statistics_dir'], 'loss_ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {results_path}")
    
    # Print comparison table
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TABLE")
    logger.info("="*80)
    logger.info(f"{'Variant':<20} {'γ':>6} {'λ':>8} {'Test AUC':>10} {'Test AP':>10}")
    logger.info("-"*60)
    for name, res in results.items():
        logger.info(f"{name:<20} {res['gamma']:>6.3f} {res['lambda']:>8.4f} "
                   f"{res['test_auc']:>10.4f} {res['test_ap']:>10.4f}")
    
    return results


if __name__ == "__main__":
    main()
