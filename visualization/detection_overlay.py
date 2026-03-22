"""Drawing detection overlays on images/frames."""

from typing import Dict, List, Tuple

import cv2
import numpy as np


# Color palette for different gestures (BGR format)
GESTURE_COLORS = {
    "Hello": (0, 255, 0),     # Green
    "Help": (0, 0, 255),      # Red
    "Home": (255, 165, 0),    # Orange (BGR)
    "No": (0, 0, 200),        # Dark Red
    "Please": (255, 255, 0),  # Cyan
    "Yes": (0, 200, 0),       # Dark Green
}

DEFAULT_COLOR = (0, 255, 0)  # Green


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on a frame.

    Args:
        frame: Input frame (BGR format).
        detections: List of detection dicts with bbox, class_name, confidence.
        show_confidence: Whether to display confidence scores.
        thickness: Line thickness for bounding boxes.

    Returns:
        Frame with drawn detections.
    """
    output = frame.copy()

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(c) for c in bbox]
        class_name = det.get("class_name", "Unknown")
        confidence = det.get("confidence", 0.0)

        color = GESTURE_COLORS.get(class_name, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        if show_confidence:
            label = f"{class_name} {confidence:.0%}"
        else:
            label = class_name

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            output,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            output,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

    return output


def draw_progress_bar(
    frame: np.ndarray,
    gesture_name: str,
    progress: float,
    position: Tuple[int, int] = (10, 40),
    bar_width: int = 300,
    bar_height: int = 25,
) -> np.ndarray:
    """Draw a progress bar showing sustained detection progress.

    Args:
        frame: Input frame.
        gesture_name: Name of gesture being tracked.
        progress: Progress ratio (0.0 to 1.0).
        position: Top-left position of the bar.
        bar_width: Width of the progress bar.
        bar_height: Height of the progress bar.

    Returns:
        Frame with progress bar drawn.
    """
    output = frame.copy()
    x, y = position

    # Background
    cv2.rectangle(
        output, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1
    )

    # Progress fill
    fill_width = int(bar_width * progress)
    if progress < 0.5:
        fill_color = (0, 165, 255)  # Orange
    elif progress < 0.9:
        fill_color = (0, 255, 255)  # Yellow
    else:
        fill_color = (0, 255, 0)    # Green

    cv2.rectangle(
        output, (x, y), (x + fill_width, y + bar_height), fill_color, -1
    )

    # Border
    cv2.rectangle(
        output, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1
    )

    # Text
    label = f"Detecting: {gesture_name} ({progress:.0%})"
    cv2.putText(
        output,
        label,
        (x + 5, y + bar_height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return output


def draw_sentence_overlay(
    frame: np.ndarray,
    sentence: str,
    position: str = "bottom",
) -> np.ndarray:
    """Draw the accumulated sentence on the frame.

    Args:
        frame: Input frame.
        sentence: Sentence text to display.
        position: 'top' or 'bottom'.

    Returns:
        Frame with sentence overlay.
    """
    if not sentence:
        return frame

    output = frame.copy()
    h, w = output.shape[:2]

    if position == "bottom":
        y = h - 15
    else:
        y = 30

    # Background bar
    overlay = output.copy()
    if position == "bottom":
        cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
    else:
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

    cv2.putText(
        output,
        sentence,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return output
