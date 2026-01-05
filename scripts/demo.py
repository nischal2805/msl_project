"""Demo script for ISL Translator

Provides multiple demo modes:
1. Live camera translation
2. Video file translation
3. Batch directory translation

Usage:
    # Live camera demo
    python scripts/demo.py --checkpoint checkpoints/best_model.pt --mode live
    
    # Single video file
    python scripts/demo.py --checkpoint checkpoints/best_model.pt --mode video --input path/to/video.mp4
    
    # Batch directory
    python scripts/demo.py --checkpoint checkpoints/best_model.pt --mode batch --input path/to/videos/
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
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.translator import ISLTranslator
from src.inference.live import LiveTranslator, CameraCapture, translate_video_file


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def demo_live_camera(
    model: ISLTranslator,
    tokenizer,
    config: dict,
    device: torch.device,
    camera_id: int = 0
):
    """Run live camera translation demo.
    
    Args:
        model: Trained ISLTranslator model
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        device: Inference device
        camera_id: Camera device ID
    """
    print("\n" + "=" * 60)
    print("LIVE CAMERA TRANSLATION DEMO")
    print("=" * 60)
    print("Instructions:")
    print("  - Perform signs in front of the camera")
    print("  - Translation will appear on screen")
    print("  - Press 'q' to quit")
    print("=" * 60 + "\n")
    
    # Create live translator
    translator = LiveTranslator(
        model=model,
        tokenizer=tokenizer,
        num_frames=config['data']['num_frames'],
        image_size=config['data']['image_size'],
        device=device,
        use_ctc=False,  # Use full autoregressive decoding
        smoothing_window=3
    )
    
    # Create camera capture
    with CameraCapture(camera_id=camera_id) as camera:
        translator.run_live(
            camera=camera,
            display=True,
            inference_interval=0.5  # Translate every 0.5 seconds
        )


def demo_single_video(
    model: ISLTranslator,
    tokenizer,
    config: dict,
    device: torch.device,
    video_path: str,
    show_video: bool = True
):
    """Translate a single video file.
    
    Args:
        model: Trained ISLTranslator model
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        device: Inference device
        video_path: Path to video file
        show_video: Whether to display video
    """
    print("\n" + "=" * 60)
    print("VIDEO FILE TRANSLATION")
    print("=" * 60)
    print(f"Video: {video_path}")
    print("=" * 60 + "\n")
    
    # Translate
    text, inference_time = translate_video_file(
        model=model,
        tokenizer=tokenizer,
        video_path=video_path,
        device=device,
        num_frames=config['data']['num_frames'],
        image_size=config['data']['image_size']
    )
    
    print(f"Translation: {text}")
    print(f"Inference Time: {inference_time * 1000:.0f}ms")
    
    # Optionally show video
    if show_video:
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
                continue
            
            # Draw translation
            cv2.putText(
                frame,
                text[:60],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Video Translation", frame)
            
            key = cv2.waitKey(30)
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause
        
        cap.release()
        cv2.destroyAllWindows()
    
    return text


def demo_batch_directory(
    model: ISLTranslator,
    tokenizer,
    config: dict,
    device: torch.device,
    input_dir: str,
    output_file: str = None
):
    """Translate all videos in a directory.
    
    Args:
        model: Trained ISLTranslator model
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        device: Inference device
        input_dir: Directory containing videos
        output_file: Output file for results
    """
    print("\n" + "=" * 60)
    print("BATCH DIRECTORY TRANSLATION")
    print("=" * 60)
    print(f"Directory: {input_dir}")
    print("=" * 60 + "\n")
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
    input_path = Path(input_dir)
    video_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in video_extensions
    ]
    
    print(f"Found {len(video_files)} videos")
    
    results = []
    total_time = 0
    
    for video_path in tqdm(video_files, desc="Translating"):
        try:
            text, inf_time = translate_video_file(
                model=model,
                tokenizer=tokenizer,
                video_path=str(video_path),
                device=device,
                num_frames=config['data']['num_frames'],
                image_size=config['data']['image_size']
            )
            
            results.append({
                'file': video_path.name,
                'translation': text,
                'inference_time_ms': inf_time * 1000
            })
            total_time += inf_time
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {video_path.name}: {e}")
            results.append({
                'file': video_path.name,
                'error': str(e)
            })
    
    # Print results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    for r in results:
        if 'error' in r:
            print(f"[ERROR] {r['file']}: {r['error']}")
        else:
            print(f"\n{r['file']}:")
            print(f"  Translation: {r['translation'][:80]}...")
            print(f"  Time: {r['inference_time_ms']:.0f}ms")
    
    # Summary
    successful = [r for r in results if 'error' not in r]
    print("\n" + "=" * 60)
    print(f"Processed: {len(successful)}/{len(video_files)} videos")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Time: {total_time / len(successful) * 1000:.0f}ms per video")
    print("=" * 60)
    
    # Save results
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def demo_interactive(
    model: ISLTranslator,
    tokenizer,
    config: dict,
    device: torch.device
):
    """Interactive demo with keyboard control.
    
    Args:
        model: Trained ISLTranslator model
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        device: Inference device
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    print("Commands:")
    print("  [Enter video path] - Translate video file")
    print("  'live' or 'l'      - Start live camera")
    print("  'quit' or 'q'      - Exit")
    print("=" * 60)
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'q', 'exit']:
            print("Goodbye!")
            break
        
        elif user_input.lower() in ['live', 'l']:
            demo_live_camera(model, tokenizer, config, device)
        
        elif os.path.isfile(user_input):
            demo_single_video(
                model, tokenizer, config, device,
                user_input, show_video=False
            )
        
        elif os.path.isdir(user_input):
            demo_batch_directory(model, tokenizer, config, device, user_input)
        
        else:
            print(f"[ERROR] File or directory not found: {user_input}")
            print("Enter a valid file path, 'live', or 'quit'")


def main():
    parser = argparse.ArgumentParser(description='ISL Translator Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['live', 'video', 'batch', 'interactive'],
                        help='Demo mode')
    parser.add_argument('--input', type=str, default=None,
                        help='Input video file or directory (for video/batch modes)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for batch results')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID for live mode')
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
    
    # Run appropriate demo mode
    if args.mode == 'live':
        demo_live_camera(model, tokenizer, config, device, args.camera)
    
    elif args.mode == 'video':
        if not args.input:
            print("[ERROR] --input required for video mode")
            return
        demo_single_video(model, tokenizer, config, device, args.input)
    
    elif args.mode == 'batch':
        if not args.input:
            print("[ERROR] --input required for batch mode")
            return
        demo_batch_directory(model, tokenizer, config, device, args.input, args.output)
    
    else:  # interactive
        demo_interactive(model, tokenizer, config, device)


if __name__ == '__main__':
    main()
