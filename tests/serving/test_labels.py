from fer_2013.evaluation.evaluate import EMOTION_LABELS
from fer_2013.serving.labels import EMOTIONS


def test_emotions_match_training_labels() -> None:
    assert tuple(EMOTION_LABELS) == EMOTIONS
