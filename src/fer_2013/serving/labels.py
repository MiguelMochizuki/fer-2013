"""Emotion labels, in the class-index order used at training time.

Duplicated from fer_2013.evaluation.evaluate on purpose: that module imports
torch, which the serving image does not install. A test keeps them in sync.
"""

EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
