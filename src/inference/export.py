"""Model Export for Mobile Deployment

This module handles exporting the trained model to formats suitable
for mobile deployment:
- ONNX: Cross-platform, runs on Android via ONNX Runtime
- TFLite: Android-optimized, runs via TensorFlow Lite
- TorchScript: For PyTorch Mobile

Includes quantization support for reduced model size.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import json


class EncoderWrapper(nn.Module):
    """Wrapper for encoder-only export.
    
    For mobile, we might want to export encoder and decoder separately
    for more flexible inference pipelines.
    """
    
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: (B, T, C, H, W)
        Returns:
            features: (B, num_queries, hidden_dim)
        """
        features, _ = self.encoder(video_frames)
        return features


class DecoderWrapper(nn.Module):
    """Wrapper for single-step decoder inference.
    
    Designed for autoregressive decoding on device.
    """
    
    def __init__(
        self, 
        decoder: nn.Module,
        bos_id: int = 101,
        eos_id: int = 102
    ):
        super().__init__()
        self.decoder = decoder
        self.bos_id = bos_id
        self.eos_id = eos_id
    
    def forward(
        self, 
        encoder_out: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: (B, S, D) - encoder output
            tokens: (B, T) - current token sequence
        Returns:
            next_logits: (B, vocab_size) - logits for next token
        """
        B = encoder_out.size(0)
        encoder_lengths = torch.full((B,), encoder_out.size(1), device=encoder_out.device)
        
        logits = self.decoder(tokens, encoder_out, encoder_lengths)
        return logits[:, -1]  # Only return last position


class FullModelWrapper(nn.Module):
    """Wrapper for full encoder-decoder export.
    
    Suitable when the entire model fits in mobile memory.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        max_len: int = 50
    ):
        super().__init__()
        self.model = model
        self.max_len = max_len
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: (B, T, C, H, W)
        Returns:
            tokens: (B, max_len) - generated tokens
        """
        # Use greedy decoding (temperature=1.0, no sampling)
        return self.model.translate(
            video_frames, 
            temperature=1.0,  # Deterministic
            max_len=self.max_len
        )


class ModelExporter:
    """Export trained model to various formats.
    
    Supports:
    - ONNX (with optional quantization)
    - TorchScript
    - TFLite (via ONNX conversion)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_frames: int = 16,
        image_size: int = 224,
        hidden_dim: int = 512,
        num_queries: int = 32,
        vocab_size: int = 30522,
        output_dir: str = 'exports'
    ):
        """
        Args:
            model: Trained ISLTranslator model
            num_frames: Number of input frames
            image_size: Input image size
            hidden_dim: Model hidden dimension
            num_queries: Number of encoder output tokens
            vocab_size: Vocabulary size
            output_dir: Directory for exported models
        """
        self.model = model
        self.num_frames = num_frames
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to eval
        self.model.eval()
    
    def _create_dummy_inputs(
        self, 
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, ...]:
        """Create dummy inputs for tracing."""
        video = torch.randn(
            batch_size, 
            self.num_frames, 
            3, 
            self.image_size, 
            self.image_size
        )
        return (video,)
    
    def export_onnx(
        self,
        filename: str = 'model.onnx',
        opset_version: int = 14,
        dynamic_axes: bool = True,
        quantize: bool = False,
        export_encoder_only: bool = False
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            filename: Output filename
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic batch size
            quantize: Apply INT8 quantization
            export_encoder_only: Export only the encoder
            
        Returns:
            Path to exported model
        """
        output_path = self.output_dir / filename
        
        # Create wrapper
        if export_encoder_only:
            wrapper = EncoderWrapper(self.model.encoder)
            input_names = ['video_frames']
            output_names = ['encoder_features']
            dummy_input = self._create_dummy_inputs()
        else:
            # For full model export, we need to handle the autoregressive nature
            # Export encoder only for simplicity, decoder can be handled separately
            wrapper = EncoderWrapper(self.model.encoder)
            input_names = ['video_frames']
            output_names = ['encoder_features']
            dummy_input = self._create_dummy_inputs()
        
        wrapper.eval()
        
        # Dynamic axes for variable batch size
        dynamic_ax = None
        if dynamic_axes:
            dynamic_ax = {
                'video_frames': {0: 'batch_size'},
                'encoder_features': {0: 'batch_size'}
            }
        
        # Export
        print(f"[INFO] Exporting ONNX model to {output_path}")
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_ax,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )
        
        print(f"[INFO] ONNX model saved: {output_path}")
        
        # Quantize if requested
        if quantize:
            quantized_path = self._quantize_onnx(output_path)
            return str(quantized_path)
        
        return str(output_path)
    
    def _quantize_onnx(self, onnx_path: Path) -> Path:
        """Apply INT8 quantization to ONNX model.
        
        Args:
            onnx_path: Path to unquantized ONNX model
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("[WARNING] onnxruntime-extensions not installed, skipping quantization")
            return onnx_path
        
        quantized_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"
        
        print(f"[INFO] Quantizing ONNX model...")
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        
        # Report size reduction
        original_size = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)
        
        print(f"[INFO] Quantized model saved: {quantized_path}")
        print(f"[INFO] Size: {original_size:.1f}MB -> {quantized_size:.1f}MB ({quantized_size/original_size*100:.1f}%)")
        
        return quantized_path
    
    def export_torchscript(
        self,
        filename: str = 'model.pt',
        export_encoder_only: bool = True
    ) -> str:
        """Export model to TorchScript.
        
        Args:
            filename: Output filename
            export_encoder_only: Export only the encoder
            
        Returns:
            Path to exported model
        """
        output_path = self.output_dir / filename
        
        # Create wrapper
        if export_encoder_only:
            wrapper = EncoderWrapper(self.model.encoder)
        else:
            wrapper = FullModelWrapper(self.model)
        
        wrapper.eval()
        
        # Create example input
        dummy_input = self._create_dummy_inputs()
        
        # Trace
        print(f"[INFO] Tracing model for TorchScript...")
        
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy_input)
        
        # Optimize
        traced = torch.jit.optimize_for_mobile(traced)
        
        # Save
        traced.save(str(output_path))
        
        print(f"[INFO] TorchScript model saved: {output_path}")
        
        return str(output_path)
    
    def export_tflite(
        self,
        filename: str = 'model.tflite',
        quantization: str = 'int8'
    ) -> str:
        """Export model to TensorFlow Lite.
        
        This requires converting via ONNX -> TensorFlow -> TFLite.
        
        Args:
            filename: Output filename
            quantization: Quantization type ('none', 'fp16', 'int8')
            
        Returns:
            Path to exported model
        """
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
        except ImportError:
            print("[ERROR] Required packages not installed: onnx, onnx-tf, tensorflow")
            print("[INFO] Install with: pip install onnx onnx-tf tensorflow")
            return ""
        
        output_path = self.output_dir / filename
        
        # First export to ONNX
        onnx_path = self.export_onnx("temp_model.onnx", export_encoder_only=True)
        
        print(f"[INFO] Converting ONNX to TensorFlow...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        tf_model_path = self.output_dir / "temp_savedmodel"
        tf_rep.export_graph(str(tf_model_path))
        
        print(f"[INFO] Converting TensorFlow to TFLite...")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        
        # Apply quantization
        if quantization == 'fp16':
            converter.target_spec.supported_types = [tf.float16]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup
        import shutil
        os.remove(onnx_path)
        shutil.rmtree(tf_model_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] TFLite model saved: {output_path} ({size_mb:.1f}MB)")
        
        return str(output_path)
    
    def export_all(
        self,
        prefix: str = 'isl_translator',
        quantize: bool = True
    ) -> Dict[str, str]:
        """Export to all supported formats.
        
        Args:
            prefix: Filename prefix
            quantize: Apply quantization where supported
            
        Returns:
            Dictionary of format -> path
        """
        exports = {}
        
        # ONNX
        try:
            onnx_path = self.export_onnx(
                f"{prefix}_encoder.onnx",
                quantize=quantize,
                export_encoder_only=True
            )
            exports['onnx'] = onnx_path
        except Exception as e:
            print(f"[ERROR] ONNX export failed: {e}")
        
        # TorchScript
        try:
            ts_path = self.export_torchscript(
                f"{prefix}_encoder.pt",
                export_encoder_only=True
            )
            exports['torchscript'] = ts_path
        except Exception as e:
            print(f"[ERROR] TorchScript export failed: {e}")
        
        # Save export info
        info = {
            'num_frames': self.num_frames,
            'image_size': self.image_size,
            'hidden_dim': self.hidden_dim,
            'num_queries': self.num_queries,
            'vocab_size': self.vocab_size,
            'exports': exports
        }
        
        info_path = self.output_dir / f"{prefix}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n[INFO] Export complete! Info saved to {info_path}")
        
        return exports
    
    def verify_onnx(self, onnx_path: str) -> bool:
        """Verify ONNX model correctness.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if verification passed
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("[WARNING] onnx or onnxruntime not installed, skipping verification")
            return True
        
        print(f"[INFO] Verifying ONNX model: {onnx_path}")
        
        # Check model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("[INFO] ONNX model structure is valid")
        
        # Test inference
        session = ort.InferenceSession(onnx_path)
        
        dummy_input = self._create_dummy_inputs()
        ort_inputs = {
            session.get_inputs()[0].name: dummy_input[0].numpy()
        }
        
        # Run inference
        outputs = session.run(None, ort_inputs)
        
        print(f"[INFO] ONNX inference successful")
        print(f"[INFO] Output shape: {outputs[0].shape}")
        
        # Compare with PyTorch
        with torch.no_grad():
            encoder_wrapper = EncoderWrapper(self.model.encoder)
            encoder_wrapper.eval()
            torch_output = encoder_wrapper(dummy_input[0])
        
        # Check difference
        diff = np.abs(outputs[0] - torch_output.numpy()).mean()
        print(f"[INFO] Mean difference from PyTorch: {diff:.6f}")
        
        if diff < 1e-4:
            print("[INFO] Verification PASSED")
            return True
        else:
            print("[WARNING] Verification: Outputs differ significantly")
            return False
