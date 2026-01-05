"""Evaluation script for ISL Translator

Evaluates trained model on test set with multiple metrics:
- BLEU score (1-gram to 4-gram)
- Word Error Rate (WER)
- Sample predictions

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/config.yaml
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
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.models.translator import ISLTranslator
from src.data.dataset import ISLVideoDataset, collate_fn


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_bleu(predictions: list, references: list) -> dict:
    """Compute BLEU scores.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        Dictionary with BLEU-1 to BLEU-4 scores
    """
    try:
        from sacrebleu import corpus_bleu, BLEU
        
        # sacrebleu expects list of lists for references
        bleu = corpus_bleu(predictions, [references])
        
        return {
            'bleu': bleu.score,
            'bleu_1': bleu.precisions[0],
            'bleu_2': bleu.precisions[1],
            'bleu_3': bleu.precisions[2],
            'bleu_4': bleu.precisions[3],
            'brevity_penalty': bleu.bp
        }
    except ImportError:
        print("[WARNING] sacrebleu not installed. Using simple BLEU approximation.")
        # Simple unigram precision
        total_correct = 0
        total_pred = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = set(ref.lower().split())
            total_correct += sum(1 for t in pred_tokens if t in ref_tokens)
            total_pred += len(pred_tokens)
        
        bleu_1 = (total_correct / total_pred * 100) if total_pred > 0 else 0
        return {'bleu': bleu_1, 'bleu_1': bleu_1}


def compute_wer(predictions: list, references: list) -> float:
    """Compute Word Error Rate.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        WER as percentage
    """
    try:
        import jiwer
        return jiwer.wer(references, predictions) * 100
    except ImportError:
        # Simple Levenshtein-based WER
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]
        
        total_errors = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            total_errors += levenshtein(pred_words, ref_words)
            total_words += len(ref_words)
        
        return (total_errors / total_words * 100) if total_words > 0 else 100


@torch.no_grad()
def evaluate(
    model: ISLTranslator,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    temperature: float = 0.8,
    use_beam_search: bool = False,
    beam_size: int = 4
) -> dict:
    """Evaluate model on dataset.
    
    Args:
        model: Trained ISLTranslator model
        dataloader: Test data loader
        tokenizer: HuggingFace tokenizer
        device: Inference device
        temperature: Sampling temperature (for sampling decoding)
        use_beam_search: Use beam search instead of sampling
        beam_size: Beam size for beam search
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_references = []
    all_uids = []
    
    print("[INFO] Running evaluation...")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        video = batch['video'].to(device)
        texts = batch['texts']
        uids = batch['uids']
        
        # Generate predictions
        if use_beam_search:
            # Beam search (one sample at a time)
            batch_preds = []
            for i in range(video.size(0)):
                tokens, _ = model.translate_beam(
                    video[i:i+1],
                    beam_size=beam_size
                )
                pred_text = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
                batch_preds.append(pred_text)
        else:
            # Sampling
            tokens = model.translate(video, temperature=temperature)
            batch_preds = [
                tokenizer.decode(t.tolist(), skip_special_tokens=True)
                for t in tokens
            ]
        
        all_predictions.extend(batch_preds)
        all_references.extend(texts)
        all_uids.extend(uids)
    
    # Compute metrics
    print("[INFO] Computing metrics...")
    
    bleu_scores = compute_bleu(all_predictions, all_references)
    wer = compute_wer(all_predictions, all_references)
    
    # Compute average lengths
    avg_pred_len = sum(len(p.split()) for p in all_predictions) / len(all_predictions)
    avg_ref_len = sum(len(r.split()) for r in all_references) / len(all_references)
    
    metrics = {
        **bleu_scores,
        'wer': wer,
        'avg_pred_length': avg_pred_len,
        'avg_ref_length': avg_ref_len,
        'num_samples': len(all_predictions)
    }
    
    return metrics, all_predictions, all_references, all_uids


def print_sample_predictions(
    predictions: list,
    references: list,
    uids: list,
    num_samples: int = 10
):
    """Print sample predictions for manual inspection."""
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    indices = list(range(min(num_samples, len(predictions))))
    
    for i in indices:
        print(f"\n[Sample {i+1}] (uid: {uids[i]})")
        print(f"  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ISL Translator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--beam-search', action='store_true',
                        help='Use beam search instead of sampling')
    parser.add_argument('--beam-size', type=int, default=4,
                        help='Beam size for beam search')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results JSON')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of sample predictions to display')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load configuration
    print(f"[INFO] Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Tokenizer
    print(f"[INFO] Loading tokenizer: {config['tokenizer']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    
    # Load model
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model, checkpoint = ISLTranslator.load_checkpoint(args.checkpoint, device)
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create dataset
    eval_ds = ISLVideoDataset(
        video_dir=config['data']['video_dir'],
        csv_path=config['data']['csv_path'],
        split=args.split,
        tokenizer=tokenizer,
        num_frames=config['data']['num_frames'],
        image_size=config['data']['image_size'],
        augment=False
    )
    
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(4, config['data'].get('num_workers', 4)),
        pin_memory=True
    )
    
    # Evaluate
    metrics, predictions, references, uids = evaluate(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        use_beam_search=args.beam_search,
        beam_size=args.beam_size
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Dataset Split:      {args.split}")
    print(f"  Number of Samples:  {metrics['num_samples']}")
    print(f"  Decoding Method:    {'Beam Search' if args.beam_search else 'Sampling'}")
    print("-" * 40)
    print(f"  BLEU Score:         {metrics['bleu']:.2f}")
    if 'bleu_1' in metrics:
        print(f"  BLEU-1:             {metrics.get('bleu_1', 0):.2f}")
        print(f"  BLEU-2:             {metrics.get('bleu_2', 0):.2f}")
        print(f"  BLEU-3:             {metrics.get('bleu_3', 0):.2f}")
        print(f"  BLEU-4:             {metrics.get('bleu_4', 0):.2f}")
    print(f"  Word Error Rate:    {metrics['wer']:.2f}%")
    print("-" * 40)
    print(f"  Avg Pred Length:    {metrics['avg_pred_length']:.1f} words")
    print(f"  Avg Ref Length:     {metrics['avg_ref_length']:.1f} words")
    print("=" * 80)
    
    # Print sample predictions
    print_sample_predictions(predictions, references, uids, args.num_samples)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'metrics': metrics,
            'config': {
                'checkpoint': args.checkpoint,
                'split': args.split,
                'temperature': args.temperature,
                'beam_search': args.beam_search,
                'beam_size': args.beam_size
            },
            'predictions': [
                {'uid': u, 'reference': r, 'prediction': p}
                for u, r, p in zip(uids, references, predictions)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[INFO] Results saved to: {output_path}")


if __name__ == '__main__':
    main()
