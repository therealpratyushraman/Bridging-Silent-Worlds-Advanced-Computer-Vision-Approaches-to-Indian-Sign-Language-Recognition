"""Real-time webcam detection for ISL gesture recognition."""

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from config.settings import get_config
from inference.detector import SignLanguageDetector
from inference.gesture_to_text import GestureToTextConverter
from inference.emotion_pipeline import EmotionPipeline
from visualization.detection_overlay import draw_detections, draw_progress_bar


def run_webcam_detection(
    source: int = 0,
    model_type: str = "yolov8",
    weights_path: Optional[str] = None,
    show_emotion: bool = True,
    sustained_seconds: float = 3.0,
):
    """Run real-time webcam detection with gesture-to-text conversion.

    Args:
        source: Webcam source index (0 for default camera).
        model_type: Model type ('yolov5', 'yolov8', or 'ensemble').
        weights_path: Path to model weights.
        show_emotion: Whether to display emotion analysis.
        sustained_seconds: Seconds of sustained detection required.
    """
    cfg = get_config()

    # Initialize components
    detector = SignLanguageDetector(
        model_type=model_type,
        weights_path=weights_path,
        conf_threshold=cfg.model.confidence_threshold,
    )

    converter = GestureToTextConverter(
        sustained_seconds=sustained_seconds,
    )

    emotion = EmotionPipeline() if show_emotion else None

    # Open webcam
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.inference.webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.inference.webcam_resolution[1])

    if not cap.isOpened():
        print(f"Error: Could not open webcam source {source}")
        return

    print(f"ISL Gesture Detection started (model: {model_type})")
    print(f"Sustained detection window: {sustained_seconds}s")
    print("Press 'q' to quit, 'c' to clear sentence, 'r' to reset")

    fps_counter = 0
    fps_start = time.time()
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detector.detect(frame)
        timestamp = time.time()

        # Update gesture converter
        confirmed = converter.update(detections, timestamp)

        # Draw detections on frame
        frame = draw_detections(frame, detections)

        # Draw progress bar for current tracking
        current_gesture, progress = converter.get_progress()
        if current_gesture:
            frame = draw_progress_bar(frame, current_gesture, progress)

        # Display confirmed gesture
        if confirmed:
            emotion_text = ""
            if emotion:
                analysis = emotion.analyze_gesture(confirmed)
                emotion_text = f" [{analysis['emotion']}]"

            cv2.putText(
                frame,
                f"Confirmed: {confirmed}{emotion_text}",
                (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Display sentence
        sentence = converter.get_sentence()
        if sentence:
            cv2.putText(
                frame,
                f"Sentence: {sentence}",
                (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f}",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("ISL Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            converter.clear_sentence()
            print("Sentence cleared")
        elif key == ord("r"):
            converter.reset()
            print("Reset complete")

    cap.release()
    cv2.destroyAllWindows()

    # Print final results
    print(f"\nFinal sentence: {converter.get_sentence()}")
    print(f"Total gestures detected: {len(converter.confirmed_gestures)}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time ISL gesture detection via webcam"
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Webcam source index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        choices=["yolov5", "yolov8", "ensemble"],
        help="Detection model type",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights",
    )
    parser.add_argument(
        "--sustained-seconds",
        type=float,
        default=3.0,
        help="Seconds of sustained detection required",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Disable emotion analysis",
    )
    args = parser.parse_args()

    run_webcam_detection(
        source=args.source,
        model_type=args.model,
        weights_path=args.weights,
        show_emotion=not args.no_emotion,
        sustained_seconds=args.sustained_seconds,
    )


if __name__ == "__main__":
    main()
