"""Complete ISL Translation Model

This module combines the video encoder and text decoder into a complete
translation model. It handles:
- Training forward pass with CTC and CE losses
- Inference with various decoding strategies
- Backbone freezing/unfreezing for transfer learning
- Model checkpointing and loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from .encoder import VideoEncoder
from .decoder import TextDecoder


class ISLTranslator(nn.Module):
    """Real-time ISL to Text translator.
    
    Architecture:
        VideoEncoder (MobileNetV3 + Temporal Attention) -> TextDecoder (Transformer)
        
    Dual-head Output:
        1. CTC Head: For streaming/real-time applications
        2. CE Head: For accurate offline translation
    
    Training:
        - Uses hybrid CTC + CE loss
        - Backbone freezing for first N epochs
        - Mixed precision support
    
    Inference:
        - Greedy decoding
        - Beam search
        - Temperature sampling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary with model configuration containing:
                - encoder: encoder config
                - temporal: temporal attention config 
                - decoder: decoder config
                - vocab_size: vocabulary size
                - pad_id, bos_id, eos_id: special token IDs
        """
        super().__init__()
        self.config = config
        
        # Video Encoder
        self.encoder = VideoEncoder(
            output_dim=config['decoder']['hidden_dim'],
            num_queries=config['temporal']['num_queries'],
            pretrained=config['encoder'].get('pretrained', True),
            freeze_epochs=config['encoder'].get('freeze_epochs', 5),
            num_temporal_conv=config['temporal'].get('num_temporal_conv', 2),
            kernel_size=config['temporal'].get('kernel_size', 3),
            dropout=config['temporal'].get('dropout', 0.1)
        )
        
        # Text Decoder
        self.decoder = TextDecoder(
            vocab_size=config['vocab_size'],
            hidden_dim=config['decoder']['hidden_dim'],
            num_layers=config['decoder']['num_layers'],
            num_heads=config['decoder']['num_heads'],
            ff_dim=config['decoder']['ff_dim'],
            dropout=config['decoder']['dropout'],
            max_len=config['decoder']['max_len'],
            pad_id=config['pad_id'],
            bos_id=config['bos_id'],
            eos_id=config['eos_id']
        )
        
        # CTC head for streaming output
        self.ctc_head = nn.Sequential(
            nn.Linear(config['decoder']['hidden_dim'], config['decoder']['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['decoder']['dropout']),
            nn.Linear(config['decoder']['hidden_dim'], config['vocab_size'])
        )
        
        # Initialize CTC head
        self._init_ctc_head()
    
    def _init_ctc_head(self):
        """Initialize CTC head weights."""
        for module in self.ctc_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        video_frames: torch.Tensor, 
        targets: torch.Tensor, 
        target_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            video_frames: (B, T, C, H, W) - input video frames
            targets: (B, max_len) - target token IDs
            target_lengths: (B,) - actual target lengths
        
        Returns:
            Dictionary containing:
                - logits: (B, max_len, vocab_size) - decoder output logits
                - ctc_logits: (B, num_queries, vocab_size) - CTC head output
                - encoder_lengths: (B,) - encoder output lengths
        """
        # Encode video
        encoder_out, encoder_lengths = self.encoder(video_frames)
        
        # Decode (teacher forcing)
        logits = self.decoder(targets, encoder_out, encoder_lengths)
        
        # CTC logits (for auxiliary CTC loss)
        ctc_logits = self.ctc_head(encoder_out)
        
        return {
            'logits': logits,
            'ctc_logits': ctc_logits,
            'encoder_lengths': encoder_lengths,
            'encoder_out': encoder_out  # For visualization/debugging
        }
    
    @torch.no_grad()
    def translate(
        self, 
        video_frames: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Translate video to text tokens using sampling.
        
        Args:
            video_frames: (B, T, C, H, W) - input video frames
            max_len: Maximum output length
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Nucleus sampling
        
        Returns:
            tokens: (B, output_len) - generated token IDs
        """
        self.eval()
        
        encoder_out, encoder_lengths = self.encoder(video_frames)
        tokens = self.decoder.generate(
            encoder_out, 
            encoder_lengths,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        return tokens
    
    @torch.no_grad()
    def translate_beam(
        self,
        video_frames: torch.Tensor,
        beam_size: int = 4,
        max_len: Optional[int] = None,
        length_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Translate video using beam search (higher quality, slower).
        
        Args:
            video_frames: (1, T, C, H, W) - single video (batch size 1)
            beam_size: Number of beams
            max_len: Maximum output length
            length_penalty: Penalty for longer sequences
        
        Returns:
            tokens: (1, output_len) - generated token IDs
            score: scalar - log probability of output
        """
        self.eval()
        assert video_frames.size(0) == 1, "Beam search only supports batch size 1"
        
        encoder_out, encoder_lengths = self.encoder(video_frames)
        tokens, score = self.decoder.beam_search(
            encoder_out,
            encoder_lengths,
            beam_size=beam_size,
            max_len=max_len,
            length_penalty=length_penalty
        )
        
        return tokens, score
    
    @torch.no_grad()
    def translate_ctc(
        self, 
        video_frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast CTC-based translation (for streaming).
        
        Uses greedy CTC decoding on encoder output.
        Faster than autoregressive but less accurate.
        
        Args:
            video_frames: (B, T, C, H, W) - input video frames
        
        Returns:
            tokens: (B, output_len) - decoded token IDs (with blanks removed)
        """
        self.eval()
        
        encoder_out, _ = self.encoder(video_frames)
        ctc_logits = self.ctc_head(encoder_out)
        
        # Greedy CTC decoding
        best_path = ctc_logits.argmax(dim=-1)  # (B, num_queries)
        
        # Remove blanks and repeated tokens
        decoded = []
        for seq in best_path:
            tokens = []
            prev_token = 0  # Blank token
            for token in seq.tolist():
                if token != 0 and token != prev_token:  # Not blank and not repeat
                    tokens.append(token)
                prev_token = token
            decoded.append(tokens)
        
        # Pad sequences
        max_len = max(len(seq) for seq in decoded) if decoded else 1
        max_len = max(max_len, 1)
        
        padded = torch.zeros(len(decoded), max_len, dtype=torch.long, device=video_frames.device)
        for i, seq in enumerate(decoded):
            if seq:
                padded[i, :len(seq)] = torch.tensor(seq)
        
        return padded
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch.
        
        Handles backbone unfreezing after warmup epochs.
        
        Args:
            epoch: Current epoch number (1-indexed)
        """
        if epoch > self.encoder.freeze_epochs and self.encoder.is_frozen:
            self.encoder.unfreeze_backbone()
    
    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        encoder_counts = self.encoder.get_param_count()
        decoder_count = self.decoder.get_param_count()
        ctc_count = sum(p.numel() for p in self.ctc_head.parameters())
        
        return {
            'encoder_backbone': encoder_counts['backbone'],
            'encoder_temporal': encoder_counts['temporal'],
            'decoder': decoder_count,
            'ctc_head': ctc_count,
            'total': encoder_counts['total'] + decoder_count + ctc_count
        }
    
    def save_checkpoint(
        self, 
        path: str, 
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        best_metric: Optional[float] = None,
        **extra
    ):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            best_metric: Best validation metric (optional)
            **extra: Additional items to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'best_metric': best_metric,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint.update(extra)
        
        torch.save(checkpoint, path)
        print(f"[INFO] Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(
        cls, 
        path: str, 
        device: torch.device = torch.device('cpu'),
        strict: bool = True
    ) -> Tuple['ISLTranslator', Dict[str, Any]]:
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model to
            strict: Whether to strictly match state dict keys
        
        Returns:
            model: Loaded ISLTranslator model
            checkpoint: Full checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model from saved config
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)
        
        print(f"[INFO] Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        
        return model, checkpoint
    
    def count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
