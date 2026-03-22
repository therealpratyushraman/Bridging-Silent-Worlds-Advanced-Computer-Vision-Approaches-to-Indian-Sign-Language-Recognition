"""Emotion / sentiment classifier for gesture-context analysis.

Uses a HuggingFace Transformers sentiment pipeline to add emotional
context to recognised ISL gestures.
"""

import logging
from typing import Any, Dict, List, Optional

from .gesture_vocabulary import GESTURE_CLASSES

logger = logging.getLogger(__name__)

# Default gesture-to-emotion associations.  These represent the typical
# emotional valence when the gesture is used in everyday ISL conversation.
_DEFAULT_EMOTION_MAP: Dict[str, Dict[str, Any]] = {
    "Hello": {
        "primary_emotion": "positive",
        "emotions": ["friendly", "welcoming", "warm"],
        "valence": 0.8,
    },
    "Help": {
        "primary_emotion": "urgent",
        "emotions": ["concerned", "anxious", "hopeful"],
        "valence": -0.2,
    },
    "Home": {
        "primary_emotion": "neutral",
        "emotions": ["comfortable", "safe", "nostalgic"],
        "valence": 0.5,
    },
    "No": {
        "primary_emotion": "negative",
        "emotions": ["disagreement", "refusal", "firm"],
        "valence": -0.6,
    },
    "Please": {
        "primary_emotion": "positive",
        "emotions": ["polite", "requesting", "hopeful"],
        "valence": 0.4,
    },
    "Yes": {
        "primary_emotion": "positive",
        "emotions": ["agreement", "affirmative", "encouraging"],
        "valence": 0.7,
    },
}


class EmotionClassifier:
    """Analyse sentiment/emotion of text or gesture contexts using a
    pre-trained HuggingFace transformer model.

    Example::

        clf = EmotionClassifier()
        result = clf.classify("I really need some help right now")
        # {'label': 'NEGATIVE', 'score': 0.98, ...}

        context = clf.classify_gesture_context("Help", "I am lost")
        # {'gesture': 'Help', 'gesture_emotion': {...}, 'text_sentiment': {...}}
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ) -> None:
        """Load the HuggingFace sentiment-analysis pipeline.

        Args:
            model_name: Model identifier on the HuggingFace Hub.  The
                default is a lightweight DistilBERT model fine-tuned on
                SST-2 for binary sentiment classification.
        """
        from transformers import pipeline

        self.model_name = model_name
        logger.info("Loading emotion classifier: %s", model_name)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
        )
        logger.info("Emotion classifier ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Dict[str, Any]:
        """Return the predicted emotion label and confidence for *text*.

        Args:
            text: A sentence or short passage of English text.

        Returns:
            Dictionary with keys:
                - ``label`` (str): Predicted sentiment label (e.g.
                  ``'POSITIVE'``, ``'NEGATIVE'``).
                - ``score`` (float): Model confidence in ``[0, 1]``.
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}

        results = self._pipeline(text, truncation=True)
        top = results[0]
        return {
            "label": top["label"],
            "score": float(top["score"]),
        }

    def classify_gesture_context(
        self,
        gesture_name: str,
        sentence_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Map a gesture to its emotional context, optionally enriched
        with surrounding sentence sentiment.

        Args:
            gesture_name: Name of the recognised gesture (e.g.
                ``'Hello'``, ``'Help'``).
            sentence_context: Optional sentence that provides additional
                context for the gesture.

        Returns:
            Dictionary with:
                - ``gesture`` — gesture name.
                - ``gesture_emotion`` — default emotion map entry for the
                  gesture (or ``None`` if unknown).
                - ``text_sentiment`` — pipeline result for
                  *sentence_context* (or ``None``).
                - ``combined_valence`` — float blending the gesture's
                  default valence with the textual sentiment score.
        """
        gesture_emotion = _DEFAULT_EMOTION_MAP.get(gesture_name)

        text_sentiment: Optional[Dict[str, Any]] = None
        if sentence_context:
            text_sentiment = self.classify(sentence_context)

        # Compute a combined valence
        combined_valence: Optional[float] = None
        if gesture_emotion is not None:
            base_valence = gesture_emotion["valence"]
            if text_sentiment is not None:
                # Convert the pipeline label + score into a signed value
                text_score = text_sentiment["score"]
                if text_sentiment["label"] == "NEGATIVE":
                    text_score = -text_score
                # Blend: 60 % gesture prior, 40 % text evidence
                combined_valence = 0.6 * base_valence + 0.4 * text_score
            else:
                combined_valence = base_valence

        return {
            "gesture": gesture_name,
            "gesture_emotion": gesture_emotion,
            "text_sentiment": text_sentiment,
            "combined_valence": combined_valence,
        }

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_emotion_mapping() -> Dict[str, Dict[str, Any]]:
        """Return the default gesture-to-emotion association table.

        Returns:
            A copy of the built-in mapping from gesture names to emotion
            metadata (primary emotion, emotion list, valence).
        """
        return {k: dict(v) for k, v in _DEFAULT_EMOTION_MAP.items()}
