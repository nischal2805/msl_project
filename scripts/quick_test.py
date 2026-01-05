"""Quick Training Test Script

Runs 5 epochs on a small subset of data to verify:
1. Model can learn (loss decreases)
2. Model doesn't collapse (generates diverse outputs)
3. Train/val/test splits work correctly

Usage:
    python scripts/quick_test.py

This script uses a small subset of data for fast iteration.
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import numpy as np

from src.models.translator import ISLTranslator
from src.data.dataset import ISLVideoDataset, collate_fn
from src.training.losses import HybridCTCCELoss


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_model_collapse(predictions, tokenizer, threshold=0.8):
    """Check if model is collapsing (generating same/similar outputs).
    
    Args:
        predictions: List of generated texts
        tokenizer: Tokenizer for decoding
        threshold: If more than this fraction are identical, it's collapse
        
    Returns:
        is_collapsed: Boolean indicating if model collapsed
        diversity_score: Fraction of unique predictions
    """
    if not predictions:
        return False, 1.0
    
    # Count unique predictions
    unique = set(predictions)
    diversity_score = len(unique) / len(predictions)
    
    # Check if any single prediction dominates
    counter = Counter(predictions)
    most_common_frac = counter.most_common(1)[0][1] / len(predictions)
    
    is_collapsed = most_common_frac > threshold or diversity_score < 0.1
    
    return is_collapsed, diversity_score


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def quick_train():
    """Run quick training test."""
    print("=" * 70)
    print("ISL TRANSLATOR - QUICK TRAINING TEST")
    print("=" * 70)
    
    set_seed(42)
    
    # Config
    config_path = project_root / 'configs' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] GPU Memory: {gpu_mem:.1f} GB")
    
    # Tokenizer
    print(f"\n[INFO] Loading tokenizer: {config['tokenizer']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    
    # Create model
    print("\n[INFO] Creating model...")
    model_config = {
        'encoder': config['model']['encoder'],
        'temporal': config['model']['temporal'],
        'decoder': config['model']['decoder'],
        'vocab_size': tokenizer.vocab_size,
        'pad_id': config['tokenizer']['pad_id'],
        'bos_id': config['tokenizer']['bos_id'],
        'eos_id': config['tokenizer']['eos_id']
    }
    
    model = ISLTranslator(model_config).to(device)
    
    # Print model info
    param_counts = model.get_param_count()
    print(f"\n[INFO] Model Parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {model.count_trainable_params():,}")
    print(f"  Frozen: {model.count_frozen_params():,}")
    
    # Create datasets with SMALL subset for quick testing
    print("\n[INFO] Creating datasets...")
    SUBSET_SIZE = 500  # Small subset for quick testing
    
    train_ds = ISLVideoDataset(
        video_dir=config['data']['video_dir'],
        csv_path=config['data']['csv_path'],
        split='train',
        tokenizer=tokenizer,
        num_frames=config['data']['num_frames'],
        image_size=config['data']['image_size'],
        augment=True
    )
    
    val_ds = ISLVideoDataset(
        video_dir=config['data']['video_dir'],
        csv_path=config['data']['csv_path'],
        split='val',
        tokenizer=tokenizer,
        num_frames=config['data']['num_frames'],
        image_size=config['data']['image_size'],
        augment=False
    )
    
    print(f"\n[INFO] Full dataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")
    
    # Create subsets
    train_indices = list(range(min(SUBSET_SIZE, len(train_ds))))
    val_indices = list(range(min(SUBSET_SIZE // 5, len(val_ds))))
    
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    
    print(f"\n[INFO] Using subset for quick test:")
    print(f"  Train subset: {len(train_subset)}")
    print(f"  Val subset: {len(val_subset)}")
    
    # Small batch size for laptop GPU
    BATCH_SIZE = 4  # Small for RTX 4060
    
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer with different LRs
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters()) + list(model.ctc_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': config['training']['encoder_lr']},
        {'params': decoder_params, 'lr': config['training']['decoder_lr']}
    ], weight_decay=config['training']['weight_decay'])
    
    # Scheduler
    total_steps = len(train_loader) * 5  # 5 epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config['training']['encoder_lr'], config['training']['decoder_lr']],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Loss
    criterion = HybridCTCCELoss(
        vocab_size=tokenizer.vocab_size,
        ctc_weight=config['training']['ctc_weight'],
        label_smoothing=config['training']['label_smoothing']
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training loop
    NUM_EPOCHS = 5
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.on_epoch_start(epoch)
        
        # ============ TRAIN ============
        model.train()
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        
        for batch in pbar:
            video = batch['video'].to(device)
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(video, tokens, lengths)
                losses = criterion(
                    outputs, tokens, lengths,
                    outputs['encoder_lengths'],
                    bos_id=config['tokenizer']['bos_id'],
                    eos_id=config['tokenizer']['eos_id']
                )
            
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += losses['loss'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.3f}",
                'ce': f"{losses['ce_loss'].item():.3f}",
                'ctc': f"{losses['ctc_loss'].item():.3f}"
            })
        
        train_loss /= num_batches
        history['train_loss'].append(train_loss)
        
        # ============ VALIDATE ============
        model.eval()
        val_loss = 0
        num_val_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                video = batch['video'].to(device)
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths'].to(device)
                texts = batch['texts']
                
                # Get loss
                outputs = model(video, tokens, lengths)
                losses = criterion(
                    outputs, tokens, lengths,
                    outputs['encoder_lengths'],
                    bos_id=config['tokenizer']['bos_id'],
                    eos_id=config['tokenizer']['eos_id']
                )
                
                val_loss += losses['loss'].item()
                num_val_batches += 1
                
                # Generate predictions for collapse check
                pred_tokens = model.translate(video, temperature=0.8, max_len=50)
                
                for pred_tok, ref_text in zip(pred_tokens, texts):
                    pred_text = tokenizer.decode(pred_tok.tolist(), skip_special_tokens=True)
                    all_predictions.append(pred_text)
                    all_targets.append(ref_text)
        
        val_loss /= num_val_batches
        history['val_loss'].append(val_loss)
        
        # Check for model collapse
        is_collapsed, diversity = check_model_collapse(all_predictions, tokenizer)
        
        epoch_time = time.time() - epoch_start
        
        # ============ PRINT RESULTS ============
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} Complete ({epoch_time:.1f}s)")
        print(f"{'=' * 50}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Diversity:  {diversity:.2%} unique predictions")
        
        if is_collapsed:
            print(f"\n  ⚠️  WARNING: Model may be collapsing! Low diversity.")
        
        # Print sample predictions
        print(f"\n  Sample Predictions:")
        for i in range(min(3, len(all_predictions))):
            print(f"    Target: {all_targets[i][:60]}...")
            print(f"    Pred:   {all_predictions[i][:60]}...")
            print()
    
    # ============ FINAL ANALYSIS ============
    print("\n" + "=" * 70)
    print("TRAINING TEST COMPLETE")
    print("=" * 70)
    
    # Check if model learned
    loss_decreased = history['train_loss'][-1] < history['train_loss'][0]
    final_diversity = diversity
    
    print(f"\n[RESULTS]")
    print(f"  Initial Train Loss: {history['train_loss'][0]:.4f}")
    print(f"  Final Train Loss:   {history['train_loss'][-1]:.4f}")
    print(f"  Loss Decreased:     {'✅ Yes' if loss_decreased else '❌ No'}")
    print(f"  Final Diversity:    {final_diversity:.2%}")
    print(f"  Model Collapsed:    {'❌ Yes' if is_collapsed else '✅ No'}")
    
    # Overall assessment
    print(f"\n[ASSESSMENT]")
    if loss_decreased and not is_collapsed:
        print("  ✅ Model is learning correctly!")
        print("  ✅ Ready for full training on A100.")
    elif not loss_decreased:
        print("  ❌ Loss did not decrease - check learning rate or architecture.")
    else:
        print("  ⚠️  Model might be collapsing - check diversity and predictions.")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/quick_test.pt'
    model.save_checkpoint(checkpoint_path, epoch=NUM_EPOCHS, best_metric=val_loss)
    print(f"\n[INFO] Quick test checkpoint saved: {checkpoint_path}")
    
    return history


if __name__ == '__main__':
    quick_train()
