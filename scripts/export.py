"""Export script for mobile deployment

Exports trained model to:
- ONNX (with optional INT8 quantization)
- TorchScript (optimized for mobile)
- TFLite (for Android deployment)

Usage:
    python scripts/export.py --checkpoint checkpoints/best_model.pt --format all
    python scripts/export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize
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

from src.models.translator import ISLTranslator
from src.inference.export import ModelExporter


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Export ISL Translator for Mobile')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--format', type=str, default='all',
                        choices=['onnx', 'torchscript', 'tflite', 'all'],
                        help='Export format')
    parser.add_argument('--output-dir', type=str, default='exports',
                        help='Output directory')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply INT8 quantization')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for export (cpu recommended)')
    args = parser.parse_args()
    
    # Load configuration
    print(f"[INFO] Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Device (use CPU for export for better compatibility)
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # Load model
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model, checkpoint = ISLTranslator.load_checkpoint(args.checkpoint, device)
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create exporter
    exporter = ModelExporter(
        model=model,
        num_frames=config['data']['num_frames'],
        image_size=config['data']['image_size'],
        hidden_dim=config['model']['decoder']['hidden_dim'],
        num_queries=config['model']['temporal']['num_queries'],
        output_dir=args.output_dir
    )
    
    # Get model param count
    param_counts = model.get_param_count()
    print(f"\n[INFO] Model size:")
    print(f"  Total params: {param_counts['total']:,}")
    print(f"  FP32 size:    ~{param_counts['total'] * 4 / 1024 / 1024:.1f} MB")
    print(f"  FP16 size:    ~{param_counts['total'] * 2 / 1024 / 1024:.1f} MB")
    print(f"  INT8 size:    ~{param_counts['total'] * 1 / 1024 / 1024:.1f} MB")
    print()
    
    # Export based on format
    if args.format == 'all':
        exports = exporter.export_all(
            prefix='isl_translator',
            quantize=args.quantize
        )
    
    elif args.format == 'onnx':
        onnx_path = exporter.export_onnx(
            filename='isl_translator_encoder.onnx',
            quantize=args.quantize,
            export_encoder_only=True
        )
        
        if args.verify:
            exporter.verify_onnx(onnx_path)
    
    elif args.format == 'torchscript':
        ts_path = exporter.export_torchscript(
            filename='isl_translator_encoder.pt',
            export_encoder_only=True
        )
    
    elif args.format == 'tflite':
        tflite_path = exporter.export_tflite(
            filename='isl_translator_encoder.tflite',
            quantization='int8' if args.quantize else 'none'
        )
    
    print("\n[INFO] Export complete!")
    print(f"[INFO] Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
