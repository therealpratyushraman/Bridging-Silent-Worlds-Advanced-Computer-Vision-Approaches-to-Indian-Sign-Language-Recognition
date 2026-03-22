"""Training script for emotion classifier on ISL gesture context."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config.settings import get_config


# Default gesture-emotion training data for ISL context
GESTURE_EMOTION_DATA = [
    ("Hello", "positive", "Greeting someone warmly"),
    ("Hello", "positive", "Welcoming a friend"),
    ("Hello", "neutral", "Casual greeting"),
    ("Help", "negative", "Requesting urgent assistance"),
    ("Help", "negative", "Distress signal"),
    ("Help", "neutral", "Asking for directions"),
    ("Home", "positive", "Going back to a safe place"),
    ("Home", "positive", "Talking about family"),
    ("Home", "neutral", "Referring to a location"),
    ("No", "negative", "Refusing something"),
    ("No", "negative", "Disagreement"),
    ("No", "neutral", "Simple negation"),
    ("Please", "positive", "Polite request"),
    ("Please", "positive", "Showing courtesy"),
    ("Please", "neutral", "Formal asking"),
    ("Yes", "positive", "Agreement and approval"),
    ("Yes", "positive", "Confirmation"),
    ("Yes", "neutral", "Simple affirmation"),
]


def prepare_training_data():
    """Prepare gesture-emotion pairs for training."""
    texts = []
    labels = []
    label_map = {"positive": 0, "negative": 1, "neutral": 2}

    for gesture, emotion, context in GESTURE_EMOTION_DATA:
        texts.append(f"{gesture}: {context}")
        labels.append(label_map[emotion])

    return texts, labels, label_map


def train_emotion_model(output_dir: str = "runs/emotion"):
    """Train or fine-tune emotion classifier for ISL context.

    For the initial version, this creates a rule-based mapping
    and validates it against the training data. For production,
    this would fine-tune a transformer model.
    """
    cfg = get_config()
    os.makedirs(output_dir, exist_ok=True)

    texts, labels, label_map = prepare_training_data()
    inverse_map = {v: k for k, v in label_map.items()}

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"Training emotion classifier:")
    print(f"  Training samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Classes: {list(label_map.keys())}")

    # Build gesture-to-emotion mapping from training data
    gesture_emotions = {}
    for text, label in zip(train_texts, train_labels):
        gesture = text.split(":")[0].strip()
        emotion = inverse_map[label]
        if gesture not in gesture_emotions:
            gesture_emotions[gesture] = []
        gesture_emotions[gesture].append(emotion)

    # Determine dominant emotion per gesture
    dominant_mapping = {}
    for gesture, emotions in gesture_emotions.items():
        from collections import Counter
        counts = Counter(emotions)
        dominant_mapping[gesture] = counts.most_common(1)[0][0]

    # Validate on validation set
    predictions = []
    for text in val_texts:
        gesture = text.split(":")[0].strip()
        pred_emotion = dominant_mapping.get(gesture, "neutral")
        predictions.append(label_map[pred_emotion])

    report = classification_report(
        val_labels,
        predictions,
        target_names=list(label_map.keys()),
        output_dict=True,
    )

    print(f"\nValidation Results:")
    print(
        classification_report(
            val_labels,
            predictions,
            target_names=list(label_map.keys()),
        )
    )

    # Save mapping
    mapping_path = os.path.join(output_dir, "gesture_emotion_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(dominant_mapping, f, indent=2)
    print(f"Gesture-emotion mapping saved to: {mapping_path}")

    # Save report
    report_path = os.path.join(output_dir, "emotion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved to: {report_path}")

    return dominant_mapping, report


def main():
    parser = argparse.ArgumentParser(
        description="Train emotion classifier for ISL gestures"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/emotion",
        help="Output directory for trained model and reports",
    )
    args = parser.parse_args()

    train_emotion_model(args.output_dir)


if __name__ == "__main__":
    main()
