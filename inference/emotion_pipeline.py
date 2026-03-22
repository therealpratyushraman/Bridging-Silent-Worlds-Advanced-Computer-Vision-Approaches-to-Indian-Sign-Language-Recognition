"""Emotion analysis integration for ISL gesture context."""

from typing import Dict, List, Optional

from config.settings import get_config


class EmotionPipeline:
    """Adds emotional context to detected ISL gestures.

    Integrates with the gesture detection pipeline to provide
    sentiment/emotion analysis of detected gesture sequences.

    Args:
        model_name: HuggingFace model name for emotion classification.
    """

    def __init__(self, model_name: str = None):
        cfg = get_config()
        self.model_name = model_name or cfg.emotion.model_name
        self._classifier = None

        # Default gesture-emotion associations
        self.gesture_emotions = {
            "Hello": {"emotion": "positive", "context": "greeting"},
            "Help": {"emotion": "negative", "context": "request for assistance"},
            "Home": {"emotion": "positive", "context": "comfort and belonging"},
            "No": {"emotion": "negative", "context": "negation or refusal"},
            "Please": {"emotion": "positive", "context": "polite request"},
            "Yes": {"emotion": "positive", "context": "affirmation"},
        }

    @property
    def classifier(self):
        """Lazy-load the emotion classifier."""
        if self._classifier is None:
            from models.emotion_classifier import EmotionClassifier
            self._classifier = EmotionClassifier(self.model_name)
        return self._classifier

    def analyze_gesture(self, gesture_name: str) -> Dict:
        """Analyze emotion for a single gesture.

        Args:
            gesture_name: Name of the detected gesture.

        Returns:
            Dictionary with emotion analysis results.
        """
        # Use pre-defined mapping for known gestures
        if gesture_name in self.gesture_emotions:
            mapping = self.gesture_emotions[gesture_name]
            return {
                "gesture": gesture_name,
                "emotion": mapping["emotion"],
                "context": mapping["context"],
                "source": "mapping",
            }

        # Fall back to ML-based classification
        try:
            result = self.classifier.classify(gesture_name)
            return {
                "gesture": gesture_name,
                "emotion": result["label"],
                "confidence": result["score"],
                "source": "model",
            }
        except Exception:
            return {
                "gesture": gesture_name,
                "emotion": "neutral",
                "context": "unknown gesture",
                "source": "default",
            }

    def analyze_sequence(self, gestures: List[str]) -> Dict:
        """Analyze emotion for a sequence of gestures.

        Args:
            gestures: List of gesture names in order.

        Returns:
            Dictionary with per-gesture and overall emotion analysis.
        """
        per_gesture = [self.analyze_gesture(g) for g in gestures]

        # Determine overall sentiment
        emotions = [r["emotion"] for r in per_gesture]
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        overall = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

        return {
            "gestures": per_gesture,
            "overall_emotion": overall,
            "sequence": " ".join(gestures),
            "emotion_distribution": emotion_counts,
        }

    def get_contextual_response(self, gesture_name: str) -> str:
        """Get a contextual text response for a gesture with emotion.

        Args:
            gesture_name: Name of the detected gesture.

        Returns:
            Human-readable contextual interpretation.
        """
        analysis = self.analyze_gesture(gesture_name)
        emotion = analysis["emotion"]
        context = analysis.get("context", "")

        responses = {
            ("Hello", "positive"): "A warm greeting has been detected.",
            ("Help", "negative"): "Someone is requesting assistance.",
            ("Home", "positive"): "Referring to home - a place of comfort.",
            ("No", "negative"): "A negation or refusal is being expressed.",
            ("Please", "positive"): "A polite request is being made.",
            ("Yes", "positive"): "An affirmation or agreement is expressed.",
        }

        key = (gesture_name, emotion)
        if key in responses:
            return responses[key]

        return f"Gesture '{gesture_name}' detected with {emotion} sentiment."
