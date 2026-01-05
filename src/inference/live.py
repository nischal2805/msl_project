"""Live/Real-time Inference for ISL Translation

This module provides real-time translation from camera feed or video files.
It handles:
- Camera capture with buffering
- Sliding window inference for continuous translation
- Smooth output with temporal smoothing
- Performance optimization for real-time inference
"""

import cv2
import torch
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple, List, Callable
import threading
import queue
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CameraCapture:
    """Threaded camera capture for smooth frame acquisition.
    
    Uses a separate thread to continuously capture frames,
    preventing the main inference loop from being blocked.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 5
    ):
        """
        Args:
            camera_id: Camera device ID (0 for default webcam)
            width: Capture width
            height: Capture height
            fps: Target frames per second
            buffer_size: Number of frames to buffer
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start camera capture thread."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"[INFO] Camera started: {self.width}x{self.height} @ {self.fps}fps")
    
    def _capture_loop(self):
        """Continuous capture loop running in separate thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # If queue is full, remove oldest frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
    
    def read(self) -> Optional[np.ndarray]:
        """Read the latest frame.
        
        Returns:
            RGB frame as numpy array, or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def read_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read frame with blocking.
        
        Args:
            timeout: Maximum time to wait for frame
            
        Returns:
            RGB frame or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print("[INFO] Camera stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class LiveTranslator:
    """Real-time ISL to text translation.
    
    Uses a sliding window approach for continuous translation:
    1. Collect N frames from camera
    2. Run inference to get translation
    3. Slide window and repeat
    
    Features:
    - Temporal smoothing to reduce flickering
    - Confidence thresholding
    - CTC fallback for faster inference
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        num_frames: int = 16,
        image_size: int = 224,
        device: torch.device = None,
        use_ctc: bool = False,
        smoothing_window: int = 3,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model: Trained ISLTranslator model
            tokenizer: HuggingFace tokenizer
            num_frames: Number of frames per inference
            image_size: Input image size
            device: Inference device
            use_ctc: Use CTC decoding (faster) instead of autoregressive
            smoothing_window: Number of predictions to smooth over
            confidence_threshold: Minimum confidence for valid prediction
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.image_size = image_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ctc = use_ctc
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        
        # Move model to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=num_frames)
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Timing
        self.last_inference_time = 0
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Preprocessed frame tensor (C, H, W)
        """
        transformed = self.transform(image=frame)['image']
        return transformed
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer.
        
        Args:
            frame: RGB image as numpy array
        """
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self.frame_buffer) >= self.num_frames
    
    @torch.no_grad()
    def translate(
        self, 
        temperature: float = 0.8,
        max_len: int = 50
    ) -> Tuple[str, float]:
        """Run translation on buffered frames.
        
        Args:
            temperature: Sampling temperature
            max_len: Maximum output length
            
        Returns:
            Tuple of (translated_text, inference_time)
        """
        if not self.is_ready():
            return "", 0.0
        
        start_time = time.time()
        
        # Prepare input tensor
        frames = list(self.frame_buffer)[-self.num_frames:]
        video = torch.stack(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        
        # Run inference
        if self.use_ctc:
            tokens = self.model.translate_ctc(video)
        else:
            tokens = self.model.translate(
                video, 
                temperature=temperature,
                max_len=max_len
            )
        
        # Decode tokens to text
        text = self.tokenizer.decode(
            tokens[0].tolist(),
            skip_special_tokens=True
        )
        
        inference_time = time.time() - start_time
        self.last_inference_time = inference_time
        
        # Add to history for smoothing
        self.prediction_history.append(text)
        
        return text, inference_time
    
    def get_smoothed_prediction(self) -> str:
        """Get temporally smoothed prediction.
        
        Uses voting or longest common substring for stability.
        
        Returns:
            Smoothed prediction string
        """
        if not self.prediction_history:
            return ""
        
        if len(self.prediction_history) == 1:
            return self.prediction_history[0]
        
        # Simple voting: return most common prediction
        from collections import Counter
        votes = Counter(self.prediction_history)
        return votes.most_common(1)[0][0]
    
    def clear_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.prediction_history.clear()
    
    def run_live(
        self,
        camera: CameraCapture,
        display: bool = True,
        callback: Optional[Callable[[str, float], None]] = None,
        inference_interval: float = 0.5
    ):
        """Run live translation loop.
        
        Args:
            camera: CameraCapture instance
            display: Whether to display video window
            callback: Function to call with (text, fps) on each prediction
            inference_interval: Seconds between inferences
        """
        print("[INFO] Starting live translation (press 'q' to quit)")
        
        frame_times = deque(maxlen=30)
        last_inference = 0
        current_text = ""
        
        while True:
            frame_start = time.time()
            
            # Get frame
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Add to buffer
            self.add_frame(frame)
            
            # Run inference at interval
            if time.time() - last_inference >= inference_interval and self.is_ready():
                text, inf_time = self.translate()
                current_text = text
                last_inference = time.time()
                
                if callback:
                    callback(text, 1.0 / inf_time if inf_time > 0 else 0)
                
                print(f"\rTranslation: {text[:60]}... ({inf_time*1000:.0f}ms)", end="")
            
            # Display
            if display:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add text overlay
                cv2.putText(
                    display_frame, 
                    current_text[:50], 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Add FPS
                if frame_times:
                    fps = len(frame_times) / sum(frame_times)
                    cv2.putText(
                        display_frame, 
                        f"FPS: {fps:.1f}", 
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 200, 0), 
                        1
                    )
                
                cv2.imshow("ISL Translation", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_times.append(time.time() - frame_start)
        
        cv2.destroyAllWindows()
        print("\n[INFO] Live translation stopped")


def translate_video_file(
    model: torch.nn.Module,
    tokenizer,
    video_path: str,
    device: torch.device = None,
    num_frames: int = 16,
    image_size: int = 224
) -> Tuple[str, float]:
    """Translate a video file to text.
    
    Args:
        model: Trained ISLTranslator model
        tokenizer: HuggingFace tokenizer
        video_path: Path to video file
        device: Inference device
        num_frames: Number of frames to sample
        image_size: Input image size
        
    Returns:
        Tuple of (translated_text, inference_time)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    # Sample frame indices
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    # Transform
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(image=frame)['image']
            frames.append(frame)
        elif frames:
            frames.append(frames[-1])
        else:
            frames.append(torch.zeros(3, image_size, image_size))
    
    cap.release()
    
    # Prepare tensor
    video = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, C, H, W)
    
    # Inference
    model.eval()
    model.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        tokens = model.translate(video, temperature=0.8)
    inference_time = time.time() - start_time
    
    # Decode
    text = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
    
    return text, inference_time
