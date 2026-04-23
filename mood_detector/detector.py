from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


@dataclass(frozen=True)
class Detection:
    bbox_xywh: tuple[int, int, int, int]  # x, y, w, h
    score: float


class FaceDetector:
    """
    Fast CPU face detector using MediaPipe FaceDetector (Tasks API).
    - Optimized for a single prominent face
    - If multiple faces are present, returns the largest
    """

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        model_path: str = "assets/mediapipe/blaze_face_short_range.tflite",
        model_url: str = _DEFAULT_MODEL_URL,
    ):
        self.model_path = model_path
        self.model_url = model_url
        self._ensure_model_exists()

        # Force CPU delegate explicitly (helps avoid unintended GPU delegate selection).
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if os.path.exists(self.model_path):
            return
        urllib.request.urlretrieve(self.model_url, self.model_path)

    def close(self) -> None:
        if getattr(self, "_detector", None) is not None:
            try:
                self._detector.close()
            except RuntimeError:
                # MediaPipe may rely on thread executors that can already be shutting down
                # during interpreter teardown; ignore shutdown-time errors.
                pass
            finally:
                self._detector = None

    def detect_largest_face(
        self,
        image_bgr: np.ndarray,
        *,
        margin: float = 0.20,
    ) -> tuple[np.ndarray | None, Detection | None]:
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return None, None

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self._detector.detect(mp_image)
        if not result.detections:
            return None, None

        best = None
        best_area = -1
        best_score = 0.0

        for det in result.detections:
            bbox = det.bounding_box  # pixel coords
            area = int(bbox.width) * int(bbox.height)
            if area > best_area:
                best_area = area
                best = bbox
                # categories[0].score is the confidence
                if det.categories:
                    best_score = float(det.categories[0].score)

        if best is None:
            return None, None

        x, y, bw, bh = int(best.origin_x), int(best.origin_y), int(best.width), int(best.height)
        mx, my = int(bw * margin), int(bh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        return crop, Detection(bbox_xywh=(x1, y1, x2 - x1, y2 - y1), score=best_score)

