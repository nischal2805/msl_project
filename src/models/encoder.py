"""Video Encoder with MobileNetV3 + Temporal Conv + Attention Pooling

CRITICAL: Sign language requires temporal modeling (motion between frames).
This encoder adds 1D temporal convolutions BEFORE attention pooling to capture
motion patterns that are crucial for sign language understanding.

Architecture Flow:
    Video Frames (B, T, C, H, W)
           ↓
    MobileNetV3 (per-frame features)
           ↓
    Frame Features (B, T, 960)
           ↓
    Temporal Convolutions (motion modeling)
           ↓
    Attention Pooling (compress to fixed tokens)
           ↓
    Video Embedding (B, num_queries, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class TemporalConvBlock(nn.Module):
    """1D depthwise separable convolutions over time to capture motion patterns.
    
    Uses depthwise separable convolutions for efficiency (critical for mobile).
    Includes residual connection and pre-normalization for stable training.
    """
    
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # Pre-norm for stable training
        self.norm = nn.LayerNorm(dim)
        
        # Depthwise conv (operates on each channel independently)
        self.conv_depthwise = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim,  # Depthwise: each channel processed separately
            bias=False
        )
        
        # Pointwise conv (mixes channels)
        self.conv_pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - temporal sequence of features
        Returns:
            out: (B, T, D) - processed features with residual
        """
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # Conv expects (B, C, T) format
        x = x.transpose(1, 2)
        
        # Depthwise separable conv
        x = self.conv_depthwise(x)
        x = self.act(x)
        x = self.conv_pointwise(x)
        
        # Back to (B, T, D)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        return x + residual


class TemporalAttentionPooling(nn.Module):
    """Learnable queries to compress frame features into fixed number of tokens.
    
    This is key for converting variable-length videos into fixed-length 
    representations that the decoder can work with efficiently.
    
    Uses cross-attention where learnable query tokens attend to frame features.
    """
    
    def __init__(
        self, 
        input_dim: int = 960,      # MobileNetV3-Large output dim
        output_dim: int = 512, 
        num_queries: int = 32,     # Output sequence length
        num_heads: int = 8, 
        num_temporal_conv: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # Learnable query tokens (learned during training)
        self.queries = nn.Parameter(
            torch.randn(num_queries, output_dim) * 0.02
        )
        
        # Project input features to output dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Temporal convolutions BEFORE pooling (captures motion)
        self.temporal_convs = nn.Sequential(
            *[TemporalConvBlock(output_dim, kernel_size, dropout) 
              for _ in range(num_temporal_conv)]
        )
        
        # Cross-attention: queries attend to frame features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Post-attention layers
        self.attn_norm = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - frame features from backbone
        Returns:
            out: (B, num_queries, output_dim) - compressed video representation
        """
        B = x.size(0)
        
        # Project input to output dimension
        x = self.input_proj(x)  # (B, T, output_dim)
        
        # Apply temporal convolutions to capture motion patterns
        x = self.temporal_convs(x)  # (B, T, output_dim)
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        
        # Cross-attention: queries attend to frame features
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = self.attn_norm(queries + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(queries)
        out = self.ffn_norm(queries + ffn_out)
        
        return out


class VideoEncoder(nn.Module):
    """MobileNetV3-Large backbone + Temporal Conv + Attention Pooling.
    
    This encoder is optimized for:
    1. Mobile deployment (MobileNetV3 is efficient)
    2. Sign language (temporal convolutions capture motion)
    3. Fixed-length output (attention pooling compresses any length video)
    
    Parameters:
        ~5.4M (MobileNetV3) + ~2.1M (Temporal + Attention) = ~7.5M params
    """
    
    def __init__(
        self,
        output_dim: int = 512,
        num_queries: int = 32,
        pretrained: bool = True,
        freeze_epochs: int = 5,
        num_temporal_conv: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self._frozen = True
        self.num_queries = num_queries
        
        # MobileNetV3-Large backbone (efficient, mobile-friendly)
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        
        # Remove classifier, keep only features
        # Output will be (B, 960, 7, 7) for 224x224 input
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_dim = 960  # MobileNetV3-Large output channels
        
        # Temporal attention pooling (includes temporal conv)
        self.temporal_attn = TemporalAttentionPooling(
            input_dim=self.backbone_dim,
            output_dim=output_dim,
            num_queries=num_queries,
            num_temporal_conv=num_temporal_conv,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Initialize frozen state
        if self._frozen:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True
        print("[INFO] Backbone frozen - only temporal attention is trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        print("[INFO] Backbone unfrozen - all parameters trainable")
    
    def forward(
        self, 
        video_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_frames: (B, T, C, H, W) - video frames
                B = batch size
                T = number of frames (e.g., 16)
                C = channels (3 for RGB)
                H, W = height, width (e.g., 224)
        
        Returns:
            features: (B, num_queries, output_dim) - video embedding
            lengths: (B,) - output lengths (all same = num_queries)
        """
        B, T, C, H, W = video_frames.shape
        
        # Process all frames through CNN backbone
        # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
        frames = video_frames.view(B * T, C, H, W)
        
        # Extract spatial features
        features = self.backbone(frames)  # (B*T, 960, 7, 7)
        features = self.pool(features)    # (B*T, 960, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, 960)
        
        # Reshape back: (B*T, 960) -> (B, T, 960)
        features = features.view(B, T, -1)
        
        # Temporal modeling + attention pooling
        features = self.temporal_attn(features)  # (B, num_queries, output_dim)
        
        # All videos output same length (num_queries)
        lengths = torch.full(
            (B,), 
            self.num_queries, 
            dtype=torch.long, 
            device=features.device
        )
        
        return features, lengths
    
    @property
    def is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._frozen
    
    def get_param_count(self) -> dict:
        """Get parameter counts for each component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        temporal_params = sum(p.numel() for p in self.temporal_attn.parameters())
        
        return {
            'backbone': backbone_params,
            'temporal': temporal_params,
            'total': backbone_params + temporal_params
        }
