# Face Emotion Recognition Pipeline (Optimized for CPU)

This project provides a real-time face emotion recognition pipeline optimized for CPU efficiency.

## Features
- **Face Detection**: MediaPipe Tasks API (BlazeFace) for fast, lightweight face detection.
- **Emotion Classification**: MobileNetV2 (PyTorch) for efficient inference.
- **Real-time Performance**: Average inference time < 10ms on modern CPUs.
- **Mood Insights**: Rule-based insights based on detected emotions.

## Project Structure
- `mood_detector/`: Installable package with the full pipeline.
  - `mood_detector/detector.py`: MediaPipe Tasks face detection (largest face).
  - `mood_detector/preprocess.py`: 224x224 resize + [0,1] scaling + ImageNet normalization.
  - `mood_detector/model.py`: MobileNetV2/EfficientNet-B0 classifier head.
  - `mood_detector/inference.py`: Fast CPU inference (batch size 1).
  - `mood_detector/engine.py`: Backwards-compatible `EmotionEngine` used by Flask.
  - `mood_detector/utils.py`: Labels + mood insights.
- `app.py`: Flask UI + API endpoints (unchanged routes).
- `scripts/train_rafdb.py`: Training script (PyTorch, RAF-DB).
- `scripts/webcam_demo.py`: Real-time webcam demo.
- `train_model.py`: Compatibility shim that calls `scripts/train_rafdb.py`.

## Requirements
- Python 3.10+
- mediapipe
- torch
- torchvision
- opencv-python-headless
- numpy

## Usage

### Inference
```python
import cv2

from mood_detector.inference import EmotionInference

inference = EmotionInference(weights_path='models/raf_db_emotion_model.pth')
image = cv2.imread('face.jpg')
result = inference.predict_bgr(image)
print(result)
```

### Real-time Webcam Demo
```bash
python3 scripts/webcam_demo.py
```

### Training
```bash
python3 train_model.py
```

If you want better accuracy (recommended), use the fine-tuning recipe via env vars:

```bash
EPOCHS=40 FREEZE_EPOCHS=10 UNFREEZE_LAST_BLOCKS=6 LR_HEAD=0.001 LR_FINETUNE=0.00005 python3 scripts/train_rafdb.py
```
