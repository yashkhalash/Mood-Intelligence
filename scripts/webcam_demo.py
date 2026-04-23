import os
import sys
from pathlib import Path

import cv2

# Ensure project root is on sys.path when running as `python3 scripts/webcam_demo.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mood_detector.inference import EmotionInference, InferenceConfig


def main():
    weights = os.environ.get("WEIGHTS", os.path.join("models", "raf_db_emotion_model.pth"))
    if not os.path.exists(weights):
        weights = None

    infer = EmotionInference(weights_path=weights, cfg=InferenceConfig(model_type="mobilenet_v2", device="cpu"))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = infer.predict_bgr(frame)
        if "error" not in out:
            x, y, w, h = out["bbox"]
            label = f"{out['emotion']} ({out['confidence']:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, out["insight"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(frame, out["error"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Mood Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

