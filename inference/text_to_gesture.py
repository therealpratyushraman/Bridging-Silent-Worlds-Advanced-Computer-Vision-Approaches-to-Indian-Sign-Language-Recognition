"""Text-to-gesture conversion for ISL learning mode."""

import os
from typing import Dict, List, Optional, Tuple

from models.gesture_vocabulary import GestureVocabulary, GESTURE_CLASSES


class TextToGestureConverter:
    """Converts text input to ISL gesture sequences.

    Maps words and phrases to their corresponding ISL gesture
    representations for educational/learning purposes.

    Args:
        assets_dir: Directory containing gesture reference images/GIFs.
    """

    def __init__(self, assets_dir: str = "web/assets/gestures"):
        self.assets_dir = assets_dir
        self.vocabulary = GestureVocabulary()

    def convert(self, text: str) -> List[Dict]:
        """Convert text to a sequence of gesture representations.

        Args:
            text: Input text to convert.

        Returns:
            List of gesture dictionaries with keys:
                - word: The original word
                - gesture_id: Class ID if known gesture, None otherwise
                - gesture_name: Gesture name
                - description: How to perform the gesture
                - image_path: Path to reference image (if available)
                - found: Whether gesture was found in vocabulary
        """
        words = text.strip().split()
        gestures = []

        for word in words:
            word_clean = word.strip(".,!?;:").capitalize()
            gesture_info = self.vocabulary.get_gesture_info_by_name(word_clean)

            if gesture_info:
                image_path = self._find_gesture_image(word_clean)
                gestures.append(
                    {
                        "word": word,
                        "gesture_id": gesture_info["class_id"],
                        "gesture_name": gesture_info["name"],
                        "description": gesture_info["description"],
                        "image_path": image_path,
                        "found": True,
                    }
                )
            else:
                # Try spelling out unknown words letter by letter
                spelled = self._spell_word(word_clean)
                gestures.append(
                    {
                        "word": word,
                        "gesture_id": None,
                        "gesture_name": None,
                        "description": f"Spell out: {spelled}",
                        "image_path": None,
                        "found": False,
                        "spelled_letters": spelled,
                    }
                )

        return gestures

    def _spell_word(self, word: str) -> List[str]:
        """Spell a word using ISL fingerspelling alphabet.

        Args:
            word: Word to spell.

        Returns:
            List of letter names for fingerspelling.
        """
        letters = []
        skip_letters = {"J", "Z"}  # Not in MNIST dataset

        for char in word.upper():
            if char.isalpha() and char not in skip_letters:
                letters.append(char)
            elif char in skip_letters:
                letters.append(f"[{char}-motion]")

        return letters

    def _find_gesture_image(self, gesture_name: str) -> Optional[str]:
        """Find reference image for a gesture.

        Args:
            gesture_name: Name of the gesture.

        Returns:
            Path to image file if found, None otherwise.
        """
        if not os.path.isdir(self.assets_dir):
            return None

        for ext in [".png", ".jpg", ".gif", ".webp"]:
            path = os.path.join(self.assets_dir, f"{gesture_name.lower()}{ext}")
            if os.path.exists(path):
                return path

        return None

    def get_available_gestures(self) -> List[str]:
        """Get list of gestures available for text-to-gesture conversion.

        Returns:
            List of available gesture names.
        """
        return list(GESTURE_CLASSES.values())

    def get_gesture_details(self, gesture_name: str) -> Optional[Dict]:
        """Get detailed information about a specific gesture.

        Args:
            gesture_name: Name of the gesture.

        Returns:
            Dictionary with gesture details or None if not found.
        """
        info = self.vocabulary.get_gesture_info_by_name(gesture_name)
        if info:
            info["image_path"] = self._find_gesture_image(gesture_name)
        return info
