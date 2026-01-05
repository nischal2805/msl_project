"""Video augmentation utilities for ISL Translation

This module provides video-specific augmentations that apply consistently
across all frames in a video. This is crucial for sign language where
temporal consistency matters.
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional
import cv2


class VideoAugmentor:
    """Apply consistent augmentations across all frames in a video.
    
    The key insight is that for sign language, we need to apply the same
    spatial transformations to all frames, but we can apply some
    temporal variations (like frame dropping or speed changes).
    """
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        image_size: int = 224,
        p_flip: float = 0.5,
        p_color: float = 0.5,
        p_geometric: float = 0.3,
        p_temporal: float = 0.2
    ):
        """
        Args:
            image_size: Output image size
            p_flip: Probability of horizontal flip
            p_color: Probability of color augmentation
            p_geometric: Probability of geometric augmentation
            p_temporal: Probability of temporal augmentation
        """
        self.image_size = image_size
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_geometric = p_geometric
        self.p_temporal = p_temporal
        
        # Normalization and tensor conversion (always applied)
        self.normalize = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ToTensorV2()
        ])
    
    def _random_flip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply horizontal flip to all frames consistently."""
        if np.random.random() < self.p_flip:
            return [cv2.flip(f, 1) for f in frames]
        return frames
    
    def _random_color(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply color jitter to all frames with same parameters."""
        if np.random.random() < self.p_color:
            # Sample augmentation parameters
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            saturation = np.random.uniform(0.8, 1.2)
            
            augmented = []
            for frame in frames:
                # Convert to HSV for saturation adjustment
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                
                # Brightness and contrast
                frame = np.clip(frame.astype(np.float32) * contrast + (brightness - 1) * 128, 0, 255)
                augmented.append(frame.astype(np.uint8))
            
            return augmented
        return frames
    
    def _random_geometric(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply geometric transformations consistently."""
        if np.random.random() < self.p_geometric:
            h, w = frames[0].shape[:2]
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            # Random scale
            scale = np.random.uniform(0.9, 1.1)
            # Random translation
            tx = np.random.uniform(-0.1, 0.1) * w
            ty = np.random.uniform(-0.1, 0.1) * h
            
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            
            return [cv2.warpAffine(f, M, (w, h)) for f in frames]
        return frames
    
    def _random_temporal(
        self, 
        frames: List[np.ndarray],
        target_frames: int
    ) -> List[np.ndarray]:
        """Apply temporal augmentations like frame dropping or speed change."""
        if np.random.random() < self.p_temporal and len(frames) > target_frames:
            # Randomly sample frames (simulates speed change)
            indices = sorted(np.random.choice(len(frames), target_frames, replace=False))
            return [frames[i] for i in indices]
        return frames
    
    def __call__(
        self, 
        frames: List[np.ndarray],
        augment: bool = True
    ) -> torch.Tensor:
        """Apply augmentations to video frames.
        
        Args:
            frames: List of RGB frames as numpy arrays
            augment: Whether to apply augmentations
            
        Returns:
            Tensor of shape (T, C, H, W)
        """
        if augment:
            frames = self._random_flip(frames)
            frames = self._random_color(frames)
            frames = self._random_geometric(frames)
        
        # Apply normalization and convert to tensor
        tensors = []
        for frame in frames:
            result = self.normalize(image=frame)['image']
            tensors.append(result)
        
        return torch.stack(tensors)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Get training augmentation pipeline.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
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
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        # Cutout/dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 8,
            max_width=image_size // 8,
            min_holes=1,
            fill_value=0,
            p=0.1
        ),
        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Get validation/test augmentation pipeline (minimal augmentations).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def mixup_video(
    video1: torch.Tensor,
    video2: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, float]:
    """Apply MixUp augmentation to videos.
    
    Args:
        video1: First video tensor (T, C, H, W)
        video2: Second video tensor (T, C, H, W)
        alpha: Beta distribution parameter
        
    Returns:
        Mixed video and lambda value
    """
    lam = np.random.beta(alpha, alpha)
    mixed = lam * video1 + (1 - lam) * video2
    return mixed, lam


def cutmix_video(
    video1: torch.Tensor,
    video2: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, float]:
    """Apply CutMix augmentation to videos.
    
    Cuts a random box from video2 and pastes it onto video1.
    
    Args:
        video1: First video tensor (T, C, H, W)
        video2: Second video tensor (T, C, H, W)
        alpha: Beta distribution parameter
        
    Returns:
        Mixed video and lambda value (area ratio)
    """
    T, C, H, W = video1.shape
    lam = np.random.beta(alpha, alpha)
    
    # Calculate box size
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    
    # Random box position
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    
    # Apply to all frames
    mixed = video1.clone()
    mixed[:, :, y1:y2, x1:x2] = video2[:, :, y1:y2, x1:x2]
    
    # Calculate actual lambda based on cut area
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    
    return mixed, lam
