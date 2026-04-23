from __future__ import annotations

import cv2
import numpy as np
import torch


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_face_bgr_to_tensor(
    face_bgr: np.ndarray,
    *,
    size: tuple[int, int] = (224, 224),
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    CPU-efficient preprocessing:
    - crop already done upstream
    - resize to 224x224
    - convert BGR -> RGB
    - scale to [0, 1]
    - apply ImageNet normalization (pretrained backbone expectation)
    Returns tensor shape: (1, 3, 224, 224)
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        raise ValueError("Empty face image")

    face = cv2.resize(face_bgr, size, interpolation=cv2.INTER_LINEAR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    x = face.astype(np.float32) / 255.0  # [0, 1]
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW

    t = torch.from_numpy(x).unsqueeze(0)  # BCHW
    if device != "cpu":
        t = t.to(device=device, non_blocking=True)
    return t

