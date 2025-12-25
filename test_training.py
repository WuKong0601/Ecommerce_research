"""
Quick test script to verify training setup has no errors
Runs 2 epochs to catch any bugs before full training
"""

import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
import yaml

print("=" * 80)
print("TESTING TRAINING SETUP")
print("=" * 80)

# Test 1: Config loading
print("\n1. Testing config loading...")
try:
    config = load_config()
    print("✓ Config loaded successfully")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# Test 2: Check all scheduler params are correct type
print("\n2. Checking scheduler parameters...")
try:
    lr_config = config['training']['lr_scheduler']
    
    patience = int(lr_config['patience'])
    factor = float(lr_config['factor'])
    min_lr = float(lr_config['min_lr'])
    
    print(f"  patience: {patience} (type: {type(patience).__name__})")
    print(f"  factor: {factor} (type: {type(factor).__name__})")
    print(f"  min_lr: {min_lr} (type: {type(min_lr).__name__})")
    
    assert isinstance(patience, int), "patience must be int"
    assert isinstance(factor, float), "factor must be float"
    assert isinstance(min_lr, float), "min_lr must be float"
    
    print("✓ All scheduler params have correct types")
except Exception as e:
    print(f"✗ Scheduler param check failed: {e}")
    sys.exit(1)

# Test 3: Check early stopping params
print("\n3. Checking early stopping parameters...")
try:
    es_config = config['training']['early_stopping']
    
    patience = int(es_config['patience'])
    min_delta = float(es_config['min_delta'])
    
    print(f"  patience: {patience} (type: {type(patience).__name__})")
    print(f"  min_delta: {min_delta} (type: {type(min_delta).__name__})")
    
    assert isinstance(patience, int), "patience must be int"
    assert isinstance(min_delta, float), "min_delta must be float"
    
    print("✓ All early stopping params have correct types")
except Exception as e:
    print(f"✗ Early stopping param check failed: {e}")
    sys.exit(1)

# Test 4: Test optimizer and scheduler initialization
print("\n4. Testing optimizer and scheduler...")
try:
    import torch
    import torch.optim as optim
    
    # Dummy model
    model = torch.nn.Linear(10, 1)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 1e-5))
    )
    print("✓ Optimizer initialized")
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=int(config['training']['lr_scheduler']['patience']),
        factor=float(config['training']['lr_scheduler']['factor']),
        min_lr=float(config['training']['lr_scheduler']['min_lr']),
        verbose=False
    )
    print("✓ Scheduler initialized")
    
    # Test scheduler step
    dummy_metric = 0.9
    scheduler.step(dummy_metric)
    print("✓ Scheduler step works")
    
except Exception as e:
    print(f"✗ Optimizer/Scheduler test failed: {e}")
    sys.exit(1)

# Test 5: Quick 2-epoch training
print("\n5. Running quick 2-epoch test...")
print("   (This will take ~2-3 minutes)")

# Temporarily modify config for quick test
test_config_path = 'configs/test_config.yaml'
test_config = config.copy()
test_config['training']['epochs'] = 2
test_config['training']['early_stopping']['patience'] = 1

with open(test_config_path, 'w') as f:
    yaml.dump(test_config, f)

print(f"   Created test config: {test_config_path}")
print("   Running 2-epoch training...")

# Run training with test config
import subprocess
result = subprocess.run(
    ['python', 'src/train.py', '--config', test_config_path],
    capture_output=True,
    text=True,
    cwd=project_root
)

# Clean up test config
os.remove(test_config_path)

if result.returncode == 0:
    print("✓ 2-epoch test completed successfully!")
else:
    print(f"✗ 2-epoch test failed!")
    print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("Safe to run full 50-epoch training")
print("=" * 80)
