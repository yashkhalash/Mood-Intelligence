from __future__ import annotations

import os
from typing import Any

import cv2

from .inference import EmotionInference, InferenceConfig
from .utils import EMOTION_CLASSES


class EmotionEngine:
    """
    Backwards-compatible engine used by the Flask app.

    It intentionally mirrors the old `inference_v3.EmotionEngine` API:
    - `process_image(image_path) -> list[dict]`
    - returns `region`, `dominant_emotion`, `emotion` (scores), `face_confidence`
    """

    def __init__(
        self,
        *,
        weights_path: str | None = None,
        model_type: str = "mobilenet_v2",
    ):
        # Default to models/raf_db_emotion_model.pth if present
        if weights_path is None:
            candidate = os.path.join("models", "raf_db_emotion_model.pth")
            weights_path = candidate if os.path.exists(candidate) else None

        self._infer = EmotionInference(
            weights_path=weights_path,
            cfg=InferenceConfig(model_type=model_type, device="cpu", torch_interop_threads=1),
        )

    def process_image(self, image_path: str) -> list[dict[str, Any]]:
        img = cv2.imread(image_path)
        if img is None:
            return []

        out = self._infer.predict_bgr(img)
        if "error" in out:
            return []

        x, y, w, h = out["bbox"]

        # Keep key names the Flask UI expects.
        return [
            {
                "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "dominant_emotion": str(out["emotion"]),
                "emotion": {k: float(out["all_scores"].get(k, 0.0)) for k in EMOTION_CLASSES},
                "face_confidence": float(out.get("face_confidence", out["confidence"])),
            }
        ]

