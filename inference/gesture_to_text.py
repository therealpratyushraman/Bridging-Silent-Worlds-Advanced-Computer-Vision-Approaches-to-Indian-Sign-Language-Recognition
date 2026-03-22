"""Gesture-to-text conversion pipeline with sustained detection."""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import get_config
from models.gesture_vocabulary import GESTURE_CLASSES


class GestureToTextConverter:
    """Converts detected gestures to text using sustained detection.

    Implements a state machine that requires a gesture to be consistently
    detected for a specified duration before confirming recognition.
    This prevents false positives from single-frame misdetections.

    Args:
        sustained_seconds: Seconds of consistent detection required.
        cooldown_seconds: Cooldown period after a gesture is confirmed.
        history_size: Number of recent detections to track.
    """

    def __init__(
        self,
        sustained_seconds: float = 3.0,
        cooldown_seconds: float = 1.0,
        history_size: int = 90,
    ):
        self.sustained_seconds = sustained_seconds
        self.cooldown_seconds = cooldown_seconds
        self.history_size = history_size

        self.detection_history = deque(maxlen=history_size)
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_confirmed_time = 0.0
        self.confirmed_gestures = []
        self.sentence_buffer = []

    def update(self, detections: List[Dict], timestamp: float = None) -> Optional[str]:
        """Process new detections and return confirmed gesture if any.

        Args:
            detections: List of detection dicts from the detector.
            timestamp: Current timestamp. If None, uses time.time().

        Returns:
            Confirmed gesture name if sustained detection threshold is met,
            None otherwise.
        """
        if timestamp is None:
            timestamp = time.time()

        # Check cooldown
        if (timestamp - self.last_confirmed_time) < self.cooldown_seconds:
            return None

        # Get the highest confidence detection
        best_detection = None
        if detections:
            best_detection = max(detections, key=lambda d: d.get("confidence", 0))

        # Record in history
        self.detection_history.append(
            {
                "timestamp": timestamp,
                "detection": best_detection,
            }
        )

        if best_detection is None:
            self._reset_tracking()
            return None

        detected_gesture = best_detection.get("class_name", "")

        # Check if same gesture as current tracking
        if detected_gesture == self.current_gesture:
            elapsed = timestamp - self.gesture_start_time
            if elapsed >= self.sustained_seconds:
                return self._confirm_gesture(detected_gesture, timestamp)
        else:
            # New gesture detected, start tracking
            self.current_gesture = detected_gesture
            self.gesture_start_time = timestamp

        return None

    def _confirm_gesture(self, gesture_name: str, timestamp: float) -> str:
        """Confirm a gesture after sustained detection.

        Args:
            gesture_name: Name of the confirmed gesture.
            timestamp: Confirmation timestamp.

        Returns:
            The confirmed gesture name.
        """
        self.confirmed_gestures.append(
            {
                "gesture": gesture_name,
                "timestamp": timestamp,
                "confidence": self._get_average_confidence(),
            }
        )
        self.sentence_buffer.append(gesture_name)
        self.last_confirmed_time = timestamp
        self._reset_tracking()

        return gesture_name

    def _reset_tracking(self):
        """Reset current gesture tracking state."""
        self.current_gesture = None
        self.gesture_start_time = None

    def _get_average_confidence(self) -> float:
        """Get average confidence of recent detections."""
        confidences = []
        for entry in self.detection_history:
            det = entry.get("detection")
            if det and det.get("class_name") == self.current_gesture:
                confidences.append(det.get("confidence", 0))

        return float(np.mean(confidences)) if confidences else 0.0

    def get_progress(self) -> Tuple[Optional[str], float]:
        """Get current detection progress.

        Returns:
            Tuple of (current_gesture, progress_ratio) where progress_ratio
            is 0.0 to 1.0 indicating how close to confirmation.
        """
        if self.current_gesture is None or self.gesture_start_time is None:
            return None, 0.0

        elapsed = time.time() - self.gesture_start_time
        progress = min(elapsed / self.sustained_seconds, 1.0)
        return self.current_gesture, progress

    def get_sentence(self) -> str:
        """Get the accumulated sentence from confirmed gestures.

        Returns:
            Space-separated string of confirmed gesture names.
        """
        return " ".join(self.sentence_buffer)

    def get_confirmed_history(self) -> List[Dict]:
        """Get list of all confirmed gestures with timestamps."""
        return list(self.confirmed_gestures)

    def clear_sentence(self):
        """Clear the sentence buffer."""
        self.sentence_buffer.clear()

    def reset(self):
        """Fully reset the converter state."""
        self.detection_history.clear()
        self._reset_tracking()
        self.last_confirmed_time = 0.0
        self.confirmed_gestures.clear()
        self.sentence_buffer.clear()
