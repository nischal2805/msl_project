"""Video Dataset for ISL Translation

This module handles loading and preprocessing of video data from the iSign dataset.
It supports:
- Video loading and frame sampling
- Data augmentation (spatial and temporal)
- Text tokenization with BERT tokenizer
- Efficient caching for faster training
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISLVideoDataset(Dataset):
    """Dataset for iSign videos with text annotations.
    
    Features:
    - Uniform frame sampling with fallback for short videos
    - Video caching to reduce I/O overhead
    - Consistent train/val/test splits using hash-based splitting
    - Configurable augmentations
    """
    
    # ImageNet normalization (used by MobileNetV3)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        video_dir: str,
        csv_path: str,
        split: str,
        tokenizer,
        num_frames: int = 16,
        image_size: int = 224,
        max_text_len: int = 100,
        augment: bool = False,
        cache_videos: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1
    ):
        """
        Args:
            video_dir: Directory containing video files (uid.mp4)
            csv_path: Path to CSV with 'uid' and 'text' columns
            split: One of 'train', 'val', 'test'
            tokenizer: HuggingFace tokenizer
            num_frames: Number of frames to sample per video
            image_size: Size to resize frames to
            max_text_len: Maximum text sequence length
            augment: Whether to apply data augmentation
            cache_videos: Whether to cache loaded videos in memory
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
        """
        self.video_dir = Path(video_dir)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.cache_videos = cache_videos
        self._video_cache: Dict[str, torch.Tensor] = {}
        
        # Token IDs (BERT special tokens)
        self.bos_id = 101  # [CLS]
        self.eos_id = 102  # [SEP]
        self.pad_id = 0    # [PAD]
        
        # Load and filter data
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        assert 'uid' in df.columns, "CSV must have 'uid' column"
        assert 'text' in df.columns, "CSV must have 'text' column"
        
        # Clean data
        df = df.dropna(subset=['uid', 'text'])
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 0]  # Remove empty texts
        
        # CRITICAL: Extract video_id from uid (format: "video_id-sequence_number")
        # Split by VIDEO_ID to prevent data leakage - same video must stay in same split!
        df['video_id'] = df['uid'].apply(lambda x: str(x).rsplit('-', 1)[0])
        
        # Hash-based splitting on VIDEO_ID (not uid) for reproducibility
        # This ensures all segments from the same video stay in the same split
        unique_videos = df['video_id'].unique()
        video_hashes = {vid: hash(str(vid)) % 100 for vid in unique_videos}
        df['_hash'] = df['video_id'].map(video_hashes)
        
        train_threshold = int(train_split * 100)
        val_threshold = train_threshold + int(val_split * 100)
        
        if split == 'train':
            self.data = df[df['_hash'] < train_threshold].reset_index(drop=True)
        elif split == 'val':
            self.data = df[
                (df['_hash'] >= train_threshold) & 
                (df['_hash'] < val_threshold)
            ].reset_index(drop=True)
        else:  # test
            self.data = df[df['_hash'] >= val_threshold].reset_index(drop=True)
        
        # Print split info
        n_videos = self.data['video_id'].nunique()
        n_samples = len(self.data)
        print(f"[{split.upper()}] {n_videos} unique videos, {n_samples} total segments")
        
        # Remove helper columns
        self.data = self.data.drop(columns=['_hash', 'video_id'])
        
        # Filter out videos that don't exist - OPTIMIZED for large datasets
        # Pre-scan directory once instead of checking each file individually
        print(f"[{split.upper()}] Scanning video directory for existing files...")
        
        existing_files = set()
        if self.video_dir.exists():
            for f in self.video_dir.iterdir():
                if f.suffix.lower() in ['.mp4', '.avi', '.webm', '.mov']:
                    # Store filename without extension
                    existing_files.add(f.stem)
        
        print(f"[{split.upper()}] Found {len(existing_files)} video files in directory")
        
        # Filter to only samples with existing videos
        initial_count = len(self.data)
        self.data = self.data[self.data['uid'].isin(existing_files)].reset_index(drop=True)
        final_count = len(self.data)
        
        if initial_count != final_count:
            print(f"[{split.upper()}] Filtered: {initial_count} -> {final_count} samples ({initial_count - final_count} missing)")
        
        print(f"[{split.upper()}] Final: {final_count} samples loaded")
        
        # Setup augmentations
        self.augment = augment and split == 'train'
        self.transform = self._build_transforms()
    
    def _build_transforms(self) -> A.Compose:
        """Build augmentation pipeline."""
        if self.augment:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                # Spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=0.5
                ),
                # Color augmentations
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.2, 
                        contrast=0.2, 
                        saturation=0.2, 
                        hue=0.1,
                        p=1.0
                    ),
                    A.RandomBrightnessContrast(p=1.0),
                ], p=0.5),
                # Noise
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
                # Normalize for ImageNet pretrained models
                A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _find_video_path(self, uid: str) -> Optional[Path]:
        """Find video file with various extensions."""
        for ext in ['.mp4', '.avi', '.webm', '.mov']:
            path = self.video_dir / f"{uid}{ext}"
            if path.exists():
                return path
        return None
    
    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Load and sample frames from video.
        
        Sampling strategy:
        - If video has more frames than needed: uniform sampling
        - If video has fewer frames: repeat last frame
        
        Args:
            video_path: Path to video file
            
        Returns:
            frames: (T, C, H, W) tensor of sampled frames
        """
        # Check cache first
        cache_key = str(video_path)
        if self.cache_videos and cache_key in self._video_cache:
            return self._video_cache[cache_key].clone()
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle edge case of very short videos
        if total_frames <= 0:
            total_frames = 1
        
        # Calculate frame indices to sample
        if total_frames < self.num_frames:
            # Repeat last frame if video is too short
            indices = list(range(total_frames))
            indices += [total_frames - 1] * (self.num_frames - total_frames)
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        
        frames = []
        prev_frame = None
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prev_frame = frame
            else:
                # Use previous frame if current frame fails
                if prev_frame is not None:
                    frame = prev_frame
                else:
                    # Create blank frame as last resort
                    frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # Apply transforms
            transformed = self.transform(image=frame)['image']
            frames.append(transformed)
        
        cap.release()
        
        result = torch.stack(frames)  # (T, C, H, W)
        
        # Cache if enabled
        if self.cache_videos:
            self._video_cache[cache_key] = result.clone()
        
        return result
    
    def _tokenize_text(self, text: str) -> tuple:
        """Tokenize text with BOS/EOS tokens.
        
        Args:
            text: Input text string
            
        Returns:
            tokens: Token IDs as list
            length: Actual length including BOS/EOS
        """
        # Tokenize without special tokens (we add them manually)
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate if needed (leave room for BOS and EOS)
        max_content_len = self.max_text_len - 2
        encoded = encoded[:max_content_len]
        
        # Add BOS and EOS
        tokens = [self.bos_id] + encoded + [self.eos_id]
        length = len(tokens)
        
        # Pad to max length
        padding_len = self.max_text_len - length
        tokens = tokens + [self.pad_id] * padding_len
        
        return tokens, length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        uid = str(row['uid'])
        text = str(row['text'])
        
        # Load video
        video_path = self._find_video_path(uid)
        if video_path is None:
            raise FileNotFoundError(f"Video not found for uid: {uid}")
        
        try:
            frames = self._load_video(video_path)
        except Exception as e:
            print(f"[WARNING] Failed to load video {uid}: {e}")
            # Return a dummy sample
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        
        # Tokenize text
        tokens, length = self._tokenize_text(text)
        
        return {
            'video': frames,                              # (T, C, H, W)
            'tokens': torch.tensor(tokens, dtype=torch.long),  # (max_text_len,)
            'length': length,                              # int
            'text': text,                                  # str (for evaluation)
            'uid': uid                                     # str (for debugging)
        }
    
    def get_sample_text(self, idx: int) -> str:
        """Get text for a sample without loading video."""
        return str(self.data.iloc[idx]['text'])


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched tensors and metadata
    """
    return {
        'video': torch.stack([x['video'] for x in batch]),       # (B, T, C, H, W)
        'tokens': torch.stack([x['tokens'] for x in batch]),     # (B, max_len)
        'lengths': torch.tensor([x['length'] for x in batch]),   # (B,)
        'texts': [x['text'] for x in batch],                      # List[str]
        'uids': [x['uid'] for x in batch]                         # List[str]
    }


def create_dataloaders(
    config: dict,
    tokenizer,
    num_workers: int = 4
) -> tuple:
    """Helper function to create train, val, test dataloaders.
    
    Args:
        config: Configuration dictionary with data settings
        tokenizer: HuggingFace tokenizer
        num_workers: Number of DataLoader workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    data_config = config['data']
    training_config = config['training']
    
    # Create datasets
    train_ds = ISLVideoDataset(
        video_dir=data_config['video_dir'],
        csv_path=data_config['csv_path'],
        split='train',
        tokenizer=tokenizer,
        num_frames=data_config['num_frames'],
        image_size=data_config['image_size'],
        augment=True
    )
    
    val_ds = ISLVideoDataset(
        video_dir=data_config['video_dir'],
        csv_path=data_config['csv_path'],
        split='val',
        tokenizer=tokenizer,
        num_frames=data_config['num_frames'],
        image_size=data_config['image_size'],
        augment=False
    )
    
    test_ds = ISLVideoDataset(
        video_dir=data_config['video_dir'],
        csv_path=data_config['csv_path'],
        split='test',
        tokenizer=tokenizer,
        num_frames=data_config['num_frames'],
        image_size=data_config['image_size'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
