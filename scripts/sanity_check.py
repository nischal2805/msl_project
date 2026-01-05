"""Quick model sanity check script.

Run this to verify the model can be created and runs correctly.

Usage:
    python scripts/sanity_check.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml


def check_model():
    """Check model creation and forward pass."""
    print("=" * 60)
    print("MODEL SANITY CHECK")
    print("=" * 60)
    
    # Load config
    config_path = project_root / 'configs' / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"\n[1] Loading configuration from {config_path}")
    
    # Create model config
    from transformers import AutoTokenizer
    
    print(f"\n[2] Loading tokenizer: {config['tokenizer']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    
    model_config = {
        'encoder': config['model']['encoder'],
        'temporal': config['model']['temporal'],
        'decoder': config['model']['decoder'],
        'vocab_size': tokenizer.vocab_size,
        'pad_id': config['tokenizer']['pad_id'],
        'bos_id': config['tokenizer']['bos_id'],
        'eos_id': config['tokenizer']['eos_id']
    }
    
    # Create model
    print("\n[3] Creating model...")
    from src.models.translator import ISLTranslator
    
    model = ISLTranslator(model_config)
    
    # Parameter counts
    param_counts = model.get_param_count()
    print(f"\n[4] Model parameters:")
    print(f"    Encoder (backbone):  {param_counts['encoder_backbone']:>12,}")
    print(f"    Encoder (temporal):  {param_counts['encoder_temporal']:>12,}")
    print(f"    Decoder:             {param_counts['decoder']:>12,}")
    print(f"    CTC Head:            {param_counts['ctc_head']:>12,}")
    print(f"    {'─' * 35}")
    print(f"    TOTAL:               {param_counts['total']:>12,}")
    print(f"    Estimated FP32:      {param_counts['total'] * 4 / 1024 / 1024:>10.1f} MB")
    print(f"    Estimated INT8:      {param_counts['total'] * 1 / 1024 / 1024:>10.1f} MB")
    
    # Test forward pass
    print("\n[5] Testing forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = config['data']['num_frames']
    image_size = config['data']['image_size']
    max_text_len = config['model']['decoder']['max_len']
    
    dummy_video = torch.randn(batch_size, num_frames, 3, image_size, image_size).to(device)
    dummy_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, max_text_len)).to(device)
    dummy_lengths = torch.randint(5, max_text_len, (batch_size,)).to(device)
    
    print(f"    Video shape:  {list(dummy_video.shape)}")
    print(f"    Tokens shape: {list(dummy_tokens.shape)}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_video, dummy_tokens, dummy_lengths)
    
    print(f"\n[6] Forward pass outputs:")
    print(f"    logits:          {list(outputs['logits'].shape)}")
    print(f"    ctc_logits:      {list(outputs['ctc_logits'].shape)}")
    print(f"    encoder_lengths: {outputs['encoder_lengths'].tolist()}")
    
    # Test translation
    print("\n[7] Testing translation (autoregressive)...")
    
    with torch.no_grad():
        tokens = model.translate(dummy_video[:1], temperature=0.8, max_len=20)
    
    print(f"    Generated tokens: {list(tokens.shape)}")
    
    decoded = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
    print(f"    Decoded text:     '{decoded[:50]}...'")
    
    # Test CTC translation
    print("\n[8] Testing CTC translation...")
    
    with torch.no_grad():
        ctc_tokens = model.translate_ctc(dummy_video[:1])
    
    print(f"    CTC tokens: {list(ctc_tokens.shape)}")
    
    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 60)
    print("\nThe model is ready for training.")
    print("Update the data paths in configs/config.yaml and run:")
    print("    python scripts/train.py --config configs/config.yaml")
    print()
    
    return True


def check_imports():
    """Check all imports work."""
    print("\n[0] Checking imports...")
    
    errors = []
    
    try:
        import torch
        print(f"    ✓ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import torchvision
        print(f"    ✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"torchvision: {e}")
    
    try:
        import transformers
        print(f"    ✓ transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"transformers: {e}")
    
    try:
        import albumentations
        print(f"    ✓ albumentations {albumentations.__version__}")
    except ImportError as e:
        errors.append(f"albumentations: {e}")
    
    try:
        import pandas
        print(f"    ✓ pandas {pandas.__version__}")
    except ImportError as e:
        errors.append(f"pandas: {e}")
    
    try:
        import cv2
        print(f"    ✓ opencv {cv2.__version__}")
    except ImportError as e:
        errors.append(f"opencv: {e}")
    
    try:
        import yaml
        print(f"    ✓ pyyaml")
    except ImportError as e:
        errors.append(f"pyyaml: {e}")
    
    try:
        import tqdm
        print(f"    ✓ tqdm")
    except ImportError as e:
        errors.append(f"tqdm: {e}")
    
    if errors:
        print("\n❌ Missing dependencies:")
        for e in errors:
            print(f"    - {e}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


if __name__ == '__main__':
    if check_imports():
        check_model()
