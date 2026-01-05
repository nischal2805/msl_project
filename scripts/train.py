"""Training script for ISL Translator

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume checkpoints/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from transformers import AutoTokenizer

from src.models.translator import ISLTranslator
from src.data.dataset import ISLVideoDataset, collate_fn
from src.training.trainer import ISLTrainer
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Create necessary directories."""
    dirs = [
        config['paths'].get('checkpoint_dir', 'checkpoints'),
        config['paths'].get('log_dir', 'logs'),
        config['paths'].get('export_dir', 'exports')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[INFO] Directory ready: {d}")


def create_model(config: dict, tokenizer) -> ISLTranslator:
    """Create model from configuration."""
    model_config = {
        'encoder': config['model']['encoder'],
        'temporal': config['model']['temporal'],
        'decoder': config['model']['decoder'],
        'vocab_size': tokenizer.vocab_size,
        'pad_id': config['tokenizer']['pad_id'],
        'bos_id': config['tokenizer']['bos_id'],
        'eos_id': config['tokenizer']['eos_id']
    }
    
    model = ISLTranslator(model_config)
    
    # Print model info
    param_counts = model.get_param_count()
    print("\n[INFO] Model Architecture:")
    print(f"  Encoder (backbone):  {param_counts['encoder_backbone']:,} params")
    print(f"  Encoder (temporal):  {param_counts['encoder_temporal']:,} params")
    print(f"  Decoder:            {param_counts['decoder']:,} params")
    print(f"  CTC Head:           {param_counts['ctc_head']:,} params")
    print(f"  TOTAL:              {param_counts['total']:,} params")
    print(f"  Estimated Size:     ~{param_counts['total'] * 4 / 1024 / 1024:.1f} MB (FP32)")
    print(f"  Estimated Size:     ~{param_counts['total'] * 1 / 1024 / 1024:.1f} MB (INT8)\n")
    
    return model


def create_dataloaders(config: dict, tokenizer) -> tuple:
    """Create data loaders."""
    data_config = config['data']
    training_config = config['training']
    
    print("[INFO] Creating datasets...")
    
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
    
    # Determine number of workers based on platform
    num_workers = data_config.get('num_workers', 4)
    if sys.platform == 'win32':
        # Windows often has issues with multiprocessing
        num_workers = min(num_workers, 4)
    
    print(f"[INFO] Using {num_workers} data loading workers")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train ISL Translator')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load configuration
    print(f"[INFO] Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Setup directories
    setup_directories(config)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[INFO] Using device: {device}")
    
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    
    # Tokenizer
    print(f"[INFO] Loading tokenizer: {config['tokenizer']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    
    # Create model
    model = create_model(config, tokenizer)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, tokenizer)
    
    # Create trainer
    trainer = ISLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tokenizer=tokenizer,
        device=device,
        checkpoint_dir=config['paths'].get('checkpoint_dir', 'checkpoints'),
        log_dir=config['paths'].get('log_dir', 'logs')
    )
    
    # Train
    resume_from = args.resume or config['training'].get('resume_from', None)
    trainer.train(resume_from=resume_from)
    
    print("\n[INFO] Training complete!")


if __name__ == '__main__':
    main()
