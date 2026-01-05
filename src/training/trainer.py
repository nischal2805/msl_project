"""Training loop and utilities for ISL Translation

This module provides a complete training pipeline with:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling with warmup
- Checkpointing and resumption
- Evaluation with BLEU score
- Logging and monitoring
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Callable
from pathlib import Path
import json
from tqdm import tqdm

from .losses import HybridCTCCELoss


class ISLTrainer:
    """Complete training pipeline for ISL translator.
    
    Features:
    - Mixed precision training with automatic scaling
    - Gradient accumulation for larger effective batch sizes
    - OneCycleLR scheduler with warmup
    - Early stopping
    - Automatic checkpointing
    - Evaluation with autoregressive decoding
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        tokenizer,
        device: torch.device,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """
        Args:
            model: ISLTranslator model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Full configuration dictionary
            tokenizer: HuggingFace tokenizer for decoding
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        train_cfg = config['training']
        self.num_epochs = train_cfg['num_epochs']
        self.gradient_accumulation = train_cfg.get('gradient_accumulation', 1)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.patience = train_cfg.get('patience', 10)
        self.use_amp = train_cfg.get('use_amp', True)
        self.save_every = train_cfg.get('save_every_n_epochs', 5)
        self.keep_last_n = train_cfg.get('keep_last_n_checkpoints', 3)
        
        # Loss function
        self.criterion = HybridCTCCELoss(
            vocab_size=tokenizer.vocab_size,
            pad_id=config['tokenizer']['pad_id'],
            ctc_weight=train_cfg['ctc_weight'],
            label_smoothing=train_cfg['label_smoothing']
        )
        
        # Optimizer with different learning rates
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[train_cfg['encoder_lr'], train_cfg['decoder_lr']],
            total_steps=total_steps,
            pct_start=train_cfg.get('warmup_ratio', 0.1)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate parameter groups."""
        train_cfg = self.config['training']
        
        # Separate encoder and decoder parameters
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = (
            list(self.model.decoder.parameters()) + 
            list(self.model.ctc_head.parameters())
        )
        
        return torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': train_cfg['encoder_lr']},
                {'params': decoder_params, 'lr': train_cfg['decoder_lr']}
            ],
            weight_decay=train_cfg['weight_decay']
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ctc_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            video = batch['video'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(video, tokens, lengths)
                losses = self.criterion(
                    outputs, 
                    tokens, 
                    lengths, 
                    outputs['encoder_lengths'],
                    bos_id=self.config['tokenizer']['bos_id'],
                    eos_id=self.config['tokenizer']['eos_id']
                )
                loss = losses['loss'] / self.gradient_accumulation
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track losses
            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_ctc_loss += losses['ctc_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'ce': f"{losses['ce_loss'].item():.4f}",
                'ctc': f"{losses['ctc_loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_ce_loss': total_ce_loss / num_batches,
            'train_ctc_loss': total_ctc_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ctc_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            video = batch['video'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            texts = batch['texts']
            
            # Forward pass for loss
            outputs = self.model(video, tokens, lengths)
            losses = self.criterion(
                outputs, 
                tokens, 
                lengths, 
                outputs['encoder_lengths'],
                bos_id=self.config['tokenizer']['bos_id'],
                eos_id=self.config['tokenizer']['eos_id']
            )
            
            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_ctc_loss += losses['ctc_loss'].item()
            num_batches += 1
            
            # Generate predictions for BLEU
            pred_tokens = self.model.translate(
                video, 
                temperature=self.config['inference'].get('temperature', 0.8)
            )
            
            for pred_tok, ref_text in zip(pred_tokens, texts):
                pred_text = self.tokenizer.decode(
                    pred_tok.tolist(), 
                    skip_special_tokens=True
                )
                all_predictions.append(pred_text)
                all_references.append(ref_text)
        
        # Compute BLEU
        bleu_score = self._compute_bleu(all_predictions, all_references)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_ce_loss': total_ce_loss / num_batches,
            'val_ctc_loss': total_ctc_loss / num_batches,
            'val_bleu': bleu_score
        }
    
    def _compute_bleu(
        self, 
        predictions: list, 
        references: list
    ) -> float:
        """Compute BLEU score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            BLEU score (0-100)
        """
        try:
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu(predictions, [references])
            return bleu.score
        except ImportError:
            # Fallback to simple token overlap
            total_correct = 0
            total_predicted = 0
            
            for pred, ref in zip(predictions, references):
                pred_tokens = set(pred.lower().split())
                ref_tokens = set(ref.lower().split())
                total_correct += len(pred_tokens & ref_tokens)
                total_predicted += len(pred_tokens)
            
            if total_predicted > 0:
                return (total_correct / total_predicted) * 100
            return 0.0
    
    def save_checkpoint(
        self, 
        filename: str,
        is_best: bool = False
    ):
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'history': self.history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"[INFO] Saved checkpoint: {path}")
        
        # Also save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"[INFO] Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint.get('history', self.history)
        
        print(f"[INFO] Resumed from epoch {self.current_epoch}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.keep_last_n:
            old_ckpt = checkpoints.pop(0)
            old_ckpt.unlink()
            print(f"[INFO] Removed old checkpoint: {old_ckpt}")
    
    def train(self, resume_from: Optional[str] = None):
        """Run full training loop.
        
        Args:
            resume_from: Optional path to checkpoint to resume from
        """
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1
        
        print(f"[INFO] Starting training from epoch {start_epoch}")
        print(f"[INFO] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"[INFO] Trainable parameters: {self.model.count_trainable_params():,}")
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Epoch start callbacks (e.g., unfreeze backbone)
            self.model.on_epoch_start(epoch)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            print(f"\nEpoch {epoch}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  Val BLEU:   {val_metrics['val_bleu']:.2f}")
            
            # Track history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['learning_rates'].append(self.scheduler.get_last_lr())
            
            # Sample prediction - show every epoch after frozen period
            if epoch >= 5 or epoch == 1:
                self._log_sample_prediction(num_samples=5)
            
            # Checkpointing
            is_best = val_metrics['val_loss'] < self.best_metric
            
            if is_best:
                self.best_metric = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.save_every == 0 or is_best:
                self.save_checkpoint(
                    f'checkpoint_epoch_{epoch}.pt',
                    is_best=is_best
                )
                self._cleanup_checkpoints()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break
        
        # Save final checkpoint and history
        self.save_checkpoint('final_model.pt')
        self._save_history()
        
        print(f"\n[INFO] Training complete!")
        print(f"[INFO] Best validation loss: {self.best_metric:.4f}")
    
    @torch.no_grad()
    def _log_sample_prediction(self, num_samples: int = 5):
        """Log sample predictions for debugging.
        
        Args:
            num_samples: Number of samples to show (5-10 recommended)
        """
        self.model.eval()
        
        batch = next(iter(self.val_loader))
        videos = batch['video'][:num_samples].to(self.device)
        ref_texts = batch['texts'][:num_samples]
        
        # Generate predictions
        pred_tokens = self.model.translate(videos, temperature=0.8)
        
        print(f"\n  Sample Predictions ({num_samples} samples):")
        print("  " + "-" * 60)
        
        for i, (pred_tok, ref_text) in enumerate(zip(pred_tokens, ref_texts)):
            pred_text = self.tokenizer.decode(
                pred_tok.tolist(), 
                skip_special_tokens=True
            )
            print(f"  [{i+1}] Target: {ref_text[:70]}...")
            print(f"      Pred:   {pred_text[:70]}...")
            print()
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'training_history.json'
        
        # Convert tensors to lists for JSON serialization
        history = {}
        for key, values in self.history.items():
            if isinstance(values[0], (list, tuple)):
                history[key] = [[float(v) for v in lr] for lr in values]
            else:
                history[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"[INFO] Saved training history: {history_path}")
