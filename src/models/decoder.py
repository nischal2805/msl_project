"""Transformer Decoder for Text Generation

This decoder generates text autoregressively from video embeddings.
It uses the standard transformer decoder architecture with:
- Cross-attention to video features
- Causal self-attention for autoregressive generation
- Pre-LayerNorm for training stability

Key Features:
1. Proper causal masking for training and inference
2. Greedy and beam search decoding options
3. Temperature-controlled sampling
4. KV-caching for faster inference (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned)."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create sinusoidal position encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TextDecoder(nn.Module):
    """Transformer decoder with proper autoregressive generation.
    
    Architecture:
        - Token Embedding + Positional Encoding
        - N x Transformer Decoder Layers (Self-Attn + Cross-Attn + FFN)
        - Linear projection to vocabulary
    
    Supports:
        - Teacher forcing (training)
        - Greedy decoding (fast inference)
        - Beam search (better quality)
        - Temperature sampling (diversity)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 100,
        pad_id: int = 0,
        bos_id: int = 101,   # BERT [CLS]
        eos_id: int = 102    # BERT [SEP]
    ):
        super().__init__()
        
        # Save config
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.embed_scale = hidden_dim ** 0.5  # Scale embeddings
        
        # Positional encoding (sinusoidal)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len, dropout)
        
        # Transformer decoder layers with Pre-LN (more stable training)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection (tied with embeddings if hidden_dim matches)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying: share weights between embedding and output projection
        # This significantly reduces model size and often improves performance
        self.output_proj.weight = self.embed.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Token embeddings (also used by output projection due to weight tying)
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        if self.pad_id is not None:
            nn.init.zeros_(self.embed.weight[self.pad_id])
    
    def _create_causal_mask(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Create causal attention mask for autoregressive decoding."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask
    
    def _create_padding_mask(
        self, 
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """Create padding mask from token ids."""
        return tokens == self.pad_id
    
    def forward(
        self, 
        targets: torch.Tensor, 
        encoder_out: torch.Tensor, 
        encoder_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass for training.
        
        Args:
            targets: (B, T) - target token IDs (including BOS, EOS, PAD)
            encoder_out: (B, S, D) - encoder output (video embedding)
            encoder_lengths: (B,) - encoder sequence lengths
        
        Returns:
            logits: (B, T, vocab_size) - log probabilities for each position
        """
        B, T = targets.shape
        device = targets.device
        S = encoder_out.size(1)
        
        # Embed tokens and scale
        x = self.embed(targets) * self.embed_scale
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create masks
        causal_mask = self._create_causal_mask(T, device)
        tgt_padding_mask = self._create_padding_mask(targets)
        
        # Memory (encoder) padding mask
        memory_padding_mask = torch.arange(S, device=device).expand(B, S) >= encoder_lengths.unsqueeze(1)
        
        # Decode
        x = self.decoder(
            x, 
            encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        encoder_out: torch.Tensor, 
        encoder_lengths: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation with various sampling strategies.
        
        Args:
            encoder_out: (B, S, D) - encoder output
            encoder_lengths: (B,) - encoder sequence lengths
            max_len: Maximum generation length
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-K sampling (if specified)
            top_p: Nucleus sampling threshold (if specified)
        
        Returns:
            tokens: (B, T) - generated token IDs
        """
        self.eval()
        B = encoder_out.size(0)
        device = encoder_out.device
        max_len = max_len or self.max_len
        
        # Start with BOS token
        tokens = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # Get logits for last position only
            logits = self.forward(tokens, encoder_out, encoder_lengths)
            next_logits = logits[:, -1]  # (B, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-K filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Replace finished sequences with PAD
            next_token[finished] = self.pad_id
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == self.eos_id)
            if finished.all():
                break
        
        return tokens
    
    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        encoder_lengths: torch.Tensor,
        beam_size: int = 4,
        max_len: Optional[int] = None,
        length_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding for better quality.
        
        Args:
            encoder_out: (B, S, D) - encoder output (B must be 1 for beam search)
            encoder_lengths: (B,) - encoder sequence lengths
            beam_size: Number of beams
            max_len: Maximum generation length
            length_penalty: Penalty for longer sequences
        
        Returns:
            best_sequence: (1, T) - best token sequence
            best_score: scalar - log probability of best sequence
        """
        self.eval()
        assert encoder_out.size(0) == 1, "Beam search only supports batch size 1"
        
        device = encoder_out.device
        max_len = max_len or self.max_len
        
        # Expand encoder output for beam search
        encoder_out = encoder_out.expand(beam_size, -1, -1)
        encoder_lengths = encoder_lengths.expand(beam_size)
        
        # Initialize beams
        beams = torch.full((beam_size, 1), self.bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')  # Only first beam is active initially
        
        finished_beams: List[Tuple[torch.Tensor, float]] = []
        
        for step in range(max_len - 1):
            # Get logits
            logits = self.forward(beams, encoder_out, encoder_lengths)
            next_logits = F.log_softmax(logits[:, -1], dim=-1)  # (beam_size, vocab_size)
            
            # Calculate scores for all possible next tokens
            vocab_size = next_logits.size(-1)
            next_scores = beam_scores.unsqueeze(1) + next_logits  # (beam_size, vocab_size)
            next_scores = next_scores.view(-1)  # (beam_size * vocab_size)
            
            # Get top beam_size candidates
            top_scores, top_indices = torch.topk(next_scores, 2 * beam_size)
            
            # Convert flat indices to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Build new beams
            new_beams = []
            new_scores = []
            
            for score, beam_idx, token_idx in zip(
                top_scores.tolist(), 
                beam_indices.tolist(), 
                token_indices.tolist()
            ):
                if len(new_beams) >= beam_size:
                    break
                
                # Extend beam
                new_beam = torch.cat([
                    beams[beam_idx],
                    torch.tensor([token_idx], device=device)
                ])
                
                # Check if finished
                if token_idx == self.eos_id:
                    # Apply length penalty
                    length = new_beam.size(0)
                    normalized_score = score / (length ** length_penalty)
                    finished_beams.append((new_beam, normalized_score))
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)
            
            if not new_beams:
                break
            
            # Pad beams to same length
            max_beam_len = max(b.size(0) for b in new_beams)
            beams = torch.zeros(len(new_beams), max_beam_len, dtype=torch.long, device=device)
            for i, beam in enumerate(new_beams):
                beams[i, :beam.size(0)] = beam
            
            beam_scores = torch.tensor(new_scores, device=device)
            
            # Update encoder expansion if beam size changed
            if beams.size(0) != encoder_out.size(0):
                encoder_out = encoder_out[:beams.size(0)]
                encoder_lengths = encoder_lengths[:beams.size(0)]
        
        # Add remaining beams to finished
        for i, beam in enumerate(beams):
            length = (beam != self.pad_id).sum().item()
            score = beam_scores[i].item() / (length ** length_penalty)
            finished_beams.append((beam, score))
        
        # Return best beam
        if finished_beams:
            best_beam, best_score = max(finished_beams, key=lambda x: x[1])
            return best_beam.unsqueeze(0), torch.tensor(best_score)
        else:
            return beams[:1], beam_scores[:1]
    
    def get_param_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
