from __future__ import annotations

from dataclasses import dataclass


# App-friendly class names (lowercase, stable keys).
EMOTION_CLASSES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


def normalize_emotion_name(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
    }
    return aliases.get(n, n)


def get_mood_insight(emotion: str, confidence: float) -> str:
    e = normalize_emotion_name(emotion)

    insights = {
        "happy": "You seem happy today! 😊 Keep spreading the joy!",
        "sad": "It's okay to feel sad. Take a deep breath and be kind to yourself.",
        "angry": "Take a moment to cool down. A short walk or deep breathing can help.",
        "surprise": "That looks like a surprise moment. What happened?",
        "fear": "Take a deep breath. You’ve got this.",
        "disgust": "Something feels off. Trust your instincts.",
        "neutral": "You seem calm and composed. Have a steady day.",
    }

    msg = insights.get(e, f"You seem to be feeling {e}. Take care.")
    if confidence < 0.55:
        return f"{msg} (Low confidence — try better lighting / a clearer face.)"
    return msg


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

