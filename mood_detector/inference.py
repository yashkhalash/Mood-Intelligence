from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .detector import FaceDetector
from .model import ModelConfig, load_emotion_model
from .preprocess import preprocess_face_bgr_to_tensor
from .utils import EMOTION_CLASSES, get_mood_insight


@dataclass(frozen=True)
class InferenceConfig:
    model_type: str = "mobilenet_v2"
    device: str = "cpu"
    torch_threads: int | None = None
    torch_interop_threads: int | None = 1


class EmotionInference:
    def __init__(self, *, weights_path: str | None, cfg: InferenceConfig = InferenceConfig()):
        self.cfg = cfg

        if self.cfg.torch_threads is not None:
            torch.set_num_threads(int(self.cfg.torch_threads))
        if self.cfg.torch_interop_threads is not None:
            torch.set_num_interop_threads(int(self.cfg.torch_interop_threads))

        self.detector = FaceDetector()
        self.model = load_emotion_model(
            weights_path=weights_path,
            cfg=ModelConfig(model_type=self.cfg.model_type, num_classes=len(EMOTION_CLASSES)),
            device=self.cfg.device,
        )

    def predict_bgr(self, image_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()

        face_crop, det = self.detector.detect_largest_face(image_bgr)
        if face_crop is None or det is None:
            return {
                "error": "No face detected",
                "insight": "I couldn't see a clear face. Try better lighting and face the camera.",
            }

        x = preprocess_face_bgr_to_tensor(face_crop, device=self.cfg.device)

        with torch.inference_mode():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)
        emotion = EMOTION_CLASSES[int(idx)]

        all_scores = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}
        t_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "emotion": emotion,
            "confidence": float(conf),
            "all_scores": all_scores,
            "insight": get_mood_insight(emotion, float(conf)),
            "inference_time_ms": t_ms,
            "bbox": det.bbox_xywh,
            "face_confidence": float(det.score),
        }

