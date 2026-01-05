"""Hyperparameter tuning utilities for ISL Translator

This script helps find optimal hyperparameters using:
- Grid search
- Random search
- Basic optimization

Usage:
    python scripts/tune.py --config configs/config.yaml --trials 20
"""

import os
import sys
import argparse
from pathlib import Path
import json
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

from src.models.translator import ISLTranslator
from src.data.dataset import ISLVideoDataset, collate_fn
from src.training.losses import HybridCTCCELoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Define hyperparameter search space
SEARCH_SPACE = {
    'encoder_lr': [1e-5, 5e-5, 1e-4, 2e-4],
    'decoder_lr': [1e-4, 2e-4, 3e-4, 5e-4],
    'weight_decay': [0.001, 0.01, 0.05],
    'ctc_weight': [0.1, 0.2, 0.3, 0.4],
    'label_smoothing': [0.0, 0.05, 0.1, 0.15],
    'dropout': [0.05, 0.1, 0.15, 0.2],
    'num_heads': [4, 8],
    'ff_dim': [1024, 2048],
    'num_temporal_conv': [1, 2, 3]
}


def sample_hyperparams(search_space: dict) -> dict:
    """Randomly sample hyperparameters from search space."""
    return {
        key: random.choice(values)
        for key, values in search_space.items()
    }


def create_config_with_params(base_config: dict, params: dict) -> dict:
    """Create config with sampled hyperparameters."""
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    # Update config with sampled params
    config['training']['encoder_lr'] = params.get('encoder_lr', config['training']['encoder_lr'])
    config['training']['decoder_lr'] = params.get('decoder_lr', config['training']['decoder_lr'])
    config['training']['weight_decay'] = params.get('weight_decay', config['training']['weight_decay'])
    config['training']['ctc_weight'] = params.get('ctc_weight', config['training']['ctc_weight'])
    config['training']['label_smoothing'] = params.get('label_smoothing', config['training']['label_smoothing'])
    
    config['model']['temporal']['dropout'] = params.get('dropout', config['model']['temporal']['dropout'])
    config['model']['decoder']['dropout'] = params.get('dropout', config['model']['decoder']['dropout'])
    config['model']['decoder']['num_heads'] = params.get('num_heads', config['model']['decoder']['num_heads'])
    config['model']['decoder']['ff_dim'] = params.get('ff_dim', config['model']['decoder']['ff_dim'])
    config['model']['temporal']['num_temporal_conv'] = params.get('num_temporal_conv', 
                                                                    config['model']['temporal'].get('num_temporal_conv', 2))
    
    return config


def train_with_params(
    params: dict,
    base_config: dict,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5
) -> float:
    """Train model with given hyperparameters and return validation loss.
    
    Args:
        params: Hyperparameter dictionary
        base_config: Base configuration
        tokenizer: HuggingFace tokenizer
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        num_epochs: Number of training epochs
        
    Returns:
        Best validation loss achieved
    """
    # Create config
    config = create_config_with_params(base_config, params)
    
    # Create model
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
    
    # Optimizer
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters()) + list(model.ctc_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': config['training']['encoder_lr']},
        {'params': decoder_params, 'lr': config['training']['decoder_lr']}
    ], weight_decay=config['training']['weight_decay'])
    
    # Loss
    criterion = HybridCTCCELoss(
        vocab_size=tokenizer.vocab_size,
        ctc_weight=config['training']['ctc_weight'],
        label_smoothing=config['training']['label_smoothing']
    )
    
    # Train
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.on_epoch_start(epoch)
        
        # Train
        model.train()
        for batch in train_loader:
            video = batch['video'].to(device)
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            outputs = model(video, tokens, lengths)
            losses = criterion(
                outputs, tokens, lengths, 
                outputs['encoder_lengths'],
                bos_id=config['tokenizer']['bos_id'],
                eos_id=config['tokenizer']['eos_id']
            )
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths'].to(device)
                
                outputs = model(video, tokens, lengths)
                losses = criterion(
                    outputs, tokens, lengths,
                    outputs['encoder_lengths'],
                    bos_id=config['tokenizer']['bos_id'],
                    eos_id=config['tokenizer']['eos_id']
                )
                val_loss += losses['loss'].item()
        
        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return best_val_loss


def run_random_search(
    base_config: dict,
    tokenizer,
    device: torch.device,
    num_trials: int = 20,
    epochs_per_trial: int = 5,
    subset_size: int = 1000
):
    """Run random hyperparameter search.
    
    Args:
        base_config: Base configuration
        tokenizer: HuggingFace tokenizer
        device: Training device
        num_trials: Number of random trials
        epochs_per_trial: Epochs per trial
        subset_size: Size of data subset for faster evaluation
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"Trials: {num_trials}")
    print(f"Epochs per trial: {epochs_per_trial}")
    print(f"Subset size: {subset_size}")
    print("=" * 60 + "\n")
    
    # Create subset data loaders for faster evaluation
    data_config = base_config['data']
    
    train_ds = ISLVideoDataset(
        video_dir=data_config['video_dir'],
        csv_path=data_config['csv_path'],
        split='train',
        tokenizer=tokenizer,
        num_frames=data_config['num_frames'],
        image_size=data_config['image_size'],
        augment=True
    )
    
    val_ds = ISLVideoDataset(
        video_dir=data_config['video_dir'],
        csv_path=data_config['csv_path'],
        split='val',
        tokenizer=tokenizer,
        num_frames=data_config['num_frames'],
        image_size=data_config['image_size'],
        augment=False
    )
    
    # Create subsets
    train_indices = list(range(min(subset_size, len(train_ds))))
    val_indices = list(range(min(subset_size // 5, len(val_ds))))
    
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Run trials
    results = []
    best_loss = float('inf')
    best_params = None
    
    for trial in range(1, num_trials + 1):
        params = sample_hyperparams(SEARCH_SPACE)
        
        print(f"\n[Trial {trial}/{num_trials}]")
        print(f"  Params: {params}")
        
        try:
            val_loss = train_with_params(
                params=params,
                base_config=base_config,
                tokenizer=tokenizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=epochs_per_trial
            )
            
            print(f"  Val Loss: {val_loss:.4f}")
            
            results.append({
                'trial': trial,
                'params': params,
                'val_loss': val_loss
            })
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                print(f"  ** NEW BEST **")
        
        except Exception as e:
            print(f"  [ERROR] Trial failed: {e}")
            results.append({
                'trial': trial,
                'params': params,
                'error': str(e)
            })
    
    # Print results
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    
    # Sort by val_loss
    successful = [r for r in results if 'val_loss' in r]
    successful.sort(key=lambda x: x['val_loss'])
    
    print("\nTop 5 configurations:")
    for i, r in enumerate(successful[:5]):
        print(f"\n{i+1}. Val Loss: {r['val_loss']:.4f}")
        for k, v in r['params'].items():
            print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"Val Loss: {best_loss:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Save results
    output_path = Path('logs') / 'hyperparam_search.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_loss': best_loss,
            'all_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return best_params, best_loss


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of random trials')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per trial')
    parser.add_argument('--subset', type=int, default=1000,
                        help='Subset size for faster evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    
    # Run search
    run_random_search(
        base_config=config,
        tokenizer=tokenizer,
        device=device,
        num_trials=args.trials,
        epochs_per_trial=args.epochs,
        subset_size=args.subset
    )


if __name__ == '__main__':
    main()
