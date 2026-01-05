"""Loss functions for ISL Translation

This module provides loss functions for training the ISL translator:
1. Cross-Entropy Loss: Standard next-token prediction loss
2. CTC Loss: For streaming/real-time applications
3. Hybrid CTC+CE Loss: Combines both for robust training

Key Considerations:
- Label smoothing for better generalization
- Proper handling of padding tokens
- CTC target preparation (excluding BOS/EOS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.
    
    Label smoothing helps prevent overconfident predictions and
    improves generalization, especially important for text generation.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        smoothing: float = 0.1, 
        ignore_index: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            smoothing: Label smoothing factor (0 = no smoothing)
            ignore_index: Index to ignore in loss computation (typically PAD)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, T, V) or (B*T, V) - model predictions
            targets: (B, T) or (B*T,) - target token IDs
            
        Returns:
            Scalar loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)
        
        # Create mask for non-padding tokens
        mask = targets != self.ignore_index
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot with smoothing
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.vocab_size - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Zero out padding targets
        smooth_targets[~mask] = 0
        
        # Compute loss
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Average over non-padding positions
        loss = loss[mask].mean() if mask.any() else loss.mean()
        
        return loss


class HybridCTCCELoss(nn.Module):
    """Combined CTC + Cross-Entropy loss for ISL translation.
    
    Why Hybrid Loss?
    1. CE loss: Good for accurate generation, provides rich gradients
    2. CTC loss: Enables streaming, handles alignment automatically
    
    The weighted combination allows the model to learn both:
    - Accurate autoregressive generation (CE)
    - Frame-to-text alignment (CTC)
    """
    
    def __init__(
        self, 
        vocab_size: int,
        pad_id: int = 0,
        blank_id: int = 0,  # CTC blank token (usually same as PAD)
        ctc_weight: float = 0.3,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            pad_id: Padding token ID
            blank_id: CTC blank token ID
            ctc_weight: Weight for CTC loss (CE weight = 1 - ctc_weight)
            label_smoothing: Label smoothing factor for CE loss
        """
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.ce_weight = 1.0 - ctc_weight
        self.pad_id = pad_id
        self.blank_id = blank_id
        
        # CTC Loss (with blank token)
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id, 
            zero_infinity=True,  # Handle inf/nan gracefully
            reduction='mean'
        )
        
        # Cross-entropy with label smoothing
        self.ce_loss = LabelSmoothingCrossEntropy(
            vocab_size=vocab_size,
            smoothing=label_smoothing,
            ignore_index=pad_id
        )
    
    def _prepare_ctc_targets(
        self, 
        targets: torch.Tensor, 
        target_lengths: torch.Tensor,
        bos_id: int = 101,
        eos_id: int = 102
    ) -> tuple:
        """Prepare targets for CTC loss.
        
        CTC targets should NOT include BOS/EOS tokens, as CTC
        handles alignment automatically.
        
        Args:
            targets: (B, T) target tokens with BOS...EOS
            target_lengths: (B,) actual lengths including BOS/EOS
            bos_id: BOS token ID
            eos_id: EOS token ID
            
        Returns:
            flat_targets: 1D tensor of concatenated targets
            ctc_lengths: (B,) lengths without BOS/EOS
        """
        batch_size = targets.size(0)
        flat_targets = []
        ctc_lengths = []
        
        for b in range(batch_size):
            # Get actual tokens (between BOS and EOS)
            length = target_lengths[b].item()
            
            # Skip BOS at position 0, skip EOS at position length-1
            # So actual content is at positions [1, length-1)
            start_idx = 1  # Skip BOS
            end_idx = length - 1  # Exclude EOS
            
            if end_idx > start_idx:
                content = targets[b, start_idx:end_idx]
                flat_targets.extend(content.tolist())
                ctc_lengths.append(end_idx - start_idx)
            else:
                # Empty sequence (just BOS+EOS)
                ctc_lengths.append(0)
        
        if flat_targets:
            flat_targets = torch.tensor(flat_targets, device=targets.device)
        else:
            flat_targets = torch.tensor([], dtype=torch.long, device=targets.device)
        
        ctc_lengths = torch.tensor(ctc_lengths, dtype=torch.long, device=targets.device)
        
        return flat_targets, ctc_lengths
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor, 
        target_lengths: torch.Tensor,
        encoder_lengths: torch.Tensor,
        bos_id: int = 101,
        eos_id: int = 102
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid CTC + CE loss.
        
        Args:
            outputs: Dictionary with:
                - 'logits': (B, T, V) decoder output logits
                - 'ctc_logits': (B, S, V) CTC head output
            targets: (B, max_len) target token IDs
            target_lengths: (B,) actual target lengths
            encoder_lengths: (B,) encoder output lengths
            bos_id: BOS token ID
            eos_id: EOS token ID
            
        Returns:
            Dictionary with:
                - 'loss': total loss
                - 'ce_loss': cross-entropy component
                - 'ctc_loss': CTC component
        """
        # ----- Cross-Entropy Loss -----
        # Shift: predict next token from previous tokens
        # Input: tokens[:-1], Target: tokens[1:]
        logits = outputs['logits'][:, :-1]  # (B, T-1, V)
        ce_targets = targets[:, 1:]          # (B, T-1)
        
        ce_loss = self.ce_loss(logits, ce_targets)
        
        # ----- CTC Loss -----
        ctc_logits = outputs['ctc_logits']  # (B, S, V)
        
        # CTC expects (T, B, C) format
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)  # (S, B, V)
        
        # Prepare CTC targets (exclude BOS/EOS)
        flat_targets, ctc_target_lengths = self._prepare_ctc_targets(
            targets, target_lengths, bos_id, eos_id
        )
        
        # Compute CTC loss
        if flat_targets.numel() > 0 and ctc_target_lengths.sum() > 0:
            ctc_loss = self.ctc_loss(
                ctc_log_probs, 
                flat_targets, 
                encoder_lengths, 
                ctc_target_lengths
            )
            
            # Handle inf/nan
            if torch.isinf(ctc_loss) or torch.isnan(ctc_loss):
                ctc_loss = torch.tensor(0.0, device=targets.device)
        else:
            ctc_loss = torch.tensor(0.0, device=targets.device)
        
        # ----- Total Loss -----
        total_loss = self.ce_weight * ce_loss + self.ctc_weight * ctc_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'ctc_loss': ctc_loss
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Useful when certain tokens appear much more frequently than others.
    Down-weights easy examples and focuses on hard ones.
    """
    
    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: Optional[float] = None,
        ignore_index: int = 0
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weight (optional)
            ignore_index: Index to ignore
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, T, V) predictions
            targets: (B, T) targets
            
        Returns:
            Scalar loss
        """
        # Flatten
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)
        
        # Mask
        mask = targets != self.ignore_index
        
        # Compute CE
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Average over non-padding
        return focal_loss[mask].mean() if mask.any() else focal_loss.mean()
