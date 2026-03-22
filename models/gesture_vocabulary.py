"""Gesture definitions and mappings for Indian Sign Language recognition."""

from typing import Dict, List, Optional, Tuple


# Primary gesture classes used in the 6-class detection model
GESTURE_CLASSES: Dict[int, str] = {
    0: "Hello",
    1: "Help",
    2: "Home",
    3: "No",
    4: "Please",
    5: "Yes",
}

# Extended ISL alphabet mapping (A-Z minus J and Z, compatible with ISL-MNIST dataset)
ISL_ALPHABET: Dict[int, str] = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "X",
    23: "Y",
}

# Detailed gesture information: (description, usage_context)
_GESTURE_INFO: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Hello",
        "description": "Open palm facing outward, fingers together, waved side to side.",
        "usage_context": "Greeting someone; initiating a conversation.",
    },
    1: {
        "name": "Help",
        "description": "Fist with thumb up placed on open palm, both hands raised upward.",
        "usage_context": "Requesting assistance; expressing need for support.",
    },
    2: {
        "name": "Home",
        "description": "Fingertips of flat hand touch the cheek near the mouth, then move to the temple.",
        "usage_context": "Referring to one's residence or a place of comfort.",
    },
    3: {
        "name": "No",
        "description": "Index and middle finger extended, snapping closed against the thumb.",
        "usage_context": "Negation; refusal; disagreement.",
    },
    4: {
        "name": "Please",
        "description": "Open palm placed on the chest, moved in a circular motion.",
        "usage_context": "Polite request; expressing courtesy.",
    },
    5: {
        "name": "Yes",
        "description": "Fist moved up and down, resembling a nodding motion.",
        "usage_context": "Affirmation; agreement; confirmation.",
    },
}

# Word-to-gesture mapping for text-to-gesture translation
_WORD_TO_GESTURE: Dict[str, int] = {
    "hello": 0,
    "hi": 0,
    "hey": 0,
    "greetings": 0,
    "help": 1,
    "assist": 1,
    "aid": 1,
    "support": 1,
    "home": 2,
    "house": 2,
    "residence": 2,
    "no": 3,
    "not": 3,
    "nope": 3,
    "negative": 3,
    "don't": 3,
    "please": 4,
    "kindly": 4,
    "request": 4,
    "yes": 5,
    "yeah": 5,
    "yep": 5,
    "affirmative": 5,
    "okay": 5,
    "ok": 5,
}


class GestureVocabulary:
    """Provides lookup and translation utilities for the ISL gesture vocabulary."""

    @staticmethod
    def get_gesture_info(cls_id: int) -> Dict[str, str]:
        """Return name, description, and usage context for a gesture class ID.

        Args:
            cls_id: Integer class ID (0-5 for the primary gesture set).

        Returns:
            Dictionary with keys 'name', 'description', 'usage_context'.

        Raises:
            KeyError: If cls_id is not in the known gesture set.
        """
        if cls_id not in _GESTURE_INFO:
            raise KeyError(
                f"Unknown gesture class ID {cls_id}. "
                f"Valid IDs: {sorted(_GESTURE_INFO.keys())}"
            )
        return dict(_GESTURE_INFO[cls_id])

    @staticmethod
    def get_all_gestures() -> List[Dict[str, object]]:
        """Return the full vocabulary list with IDs and metadata.

        Returns:
            List of dicts, each containing 'id', 'name', 'description',
            'usage_context'.
        """
        gestures = []
        for cls_id in sorted(_GESTURE_INFO.keys()):
            entry = {"id": cls_id}
            entry.update(_GESTURE_INFO[cls_id])
            gestures.append(entry)
        return gestures

    @staticmethod
    def text_to_gestures(text: str) -> List[Tuple[int, str]]:
        """Map words and phrases in *text* to a sequence of gesture IDs.

        Words that have no known gesture mapping are silently skipped.  The
        returned list preserves the order of the input tokens.

        Args:
            text: Free-form English text (e.g. "Hello please help").

        Returns:
            List of (class_id, gesture_name) tuples in input order.
        """
        tokens = text.lower().split()
        result: List[Tuple[int, str]] = []
        for token in tokens:
            # Strip common punctuation for matching
            cleaned = token.strip(".,!?;:'\"")
            if cleaned in _WORD_TO_GESTURE:
                cls_id = _WORD_TO_GESTURE[cleaned]
                result.append((cls_id, GESTURE_CLASSES[cls_id]))
        return result

    @staticmethod
    def gesture_to_text(gesture_sequence: List[int]) -> str:
        """Convert a sequence of gesture class IDs into readable text.

        Args:
            gesture_sequence: List of integer class IDs.

        Returns:
            Space-separated string of gesture names.  Unknown IDs are
            rendered as ``'[unknown:<id>]'``.
        """
        words: List[str] = []
        for cls_id in gesture_sequence:
            if cls_id in GESTURE_CLASSES:
                words.append(GESTURE_CLASSES[cls_id])
            else:
                words.append(f"[unknown:{cls_id}]")
        return " ".join(words)
