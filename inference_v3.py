"""
Legacy entrypoint kept for backwards compatibility.

Old versions used YOLOv8 + TensorFlow. The project has been refactored to a lightweight
CPU-first pipeline (MediaPipe + PyTorch) under `mood_detector/`.
"""

from mood_detector.engine import EmotionEngine


__all__ = ["EmotionEngine"]
