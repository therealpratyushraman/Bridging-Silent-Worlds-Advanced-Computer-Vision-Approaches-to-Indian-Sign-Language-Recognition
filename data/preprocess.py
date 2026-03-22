"""Preprocessing pipeline for ISL recognition datasets.

Handles letterbox resizing, MNIST-to-YOLO conversion, video frame
extraction, histogram equalization, and YOLO label file generation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Letterbox resize
# ---------------------------------------------------------------------------

def letterbox_resize(
    image: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize *image* to *target_size* preserving aspect ratio, padding the rest.

    Parameters
    ----------
    image:
        BGR or grayscale numpy array.
    target_size:
        Desired square output dimension.
    color:
        Padding colour (BGR).

    Returns
    -------
    np.ndarray
        Letterboxed image of shape ``(target_size, target_size, C)``.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# MNIST -> YOLO conversion
# ---------------------------------------------------------------------------

def convert_mnist_to_yolo(
    mnist_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 640,
) -> None:
    """Convert Sign Language MNIST CSV files to YOLO-format images and labels.

    Each 28x28 grayscale row is converted to a 3-channel image, resized to
    *target_size*, and written as a PNG.  The corresponding YOLO label file
    contains a single centred bounding box spanning the full image (class
    detection task treated as classification with a full-image box).

    Parameters
    ----------
    mnist_dir:
        Directory containing ``sign_mnist_train.csv`` and
        ``sign_mnist_test.csv``.
    output_dir:
        Root output directory.  ``images/`` and ``labels/`` subdirectories
        are created automatically.
    target_size:
        Output image dimension (square).
    """
    mnist_dir = Path(mnist_dir)
    output_dir = Path(output_dir)

    splits = {
        "train": "sign_mnist_train.csv",
        "test": "sign_mnist_test.csv",
    }

    for split_name, csv_name in splits.items():
        csv_path = mnist_dir / csv_name
        if not csv_path.exists():
            print(f"[preprocess] WARNING: {csv_path} not found — skipping.")
            continue

        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"[preprocess] Converting {csv_path.name} -> {split_name} ...")
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)  # skip header row

            for idx, row in enumerate(tqdm(reader, desc=split_name)):
                label = int(row[0])
                pixels = np.array(row[1:], dtype=np.uint8).reshape(28, 28)

                # Convert to 3-channel and resize
                image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                image = letterbox_resize(image, target_size=target_size)

                # Save image
                img_name = f"{split_name}_{idx:06d}.png"
                cv2.imwrite(str(img_dir / img_name), image)

                # YOLO label: <class> <cx> <cy> <w> <h>  (normalised)
                # Full-image bounding box
                lbl_name = f"{split_name}_{idx:06d}.txt"
                with open(lbl_dir / lbl_name, "w", encoding="utf-8") as lf:
                    lf.write(f"{label} 0.5 0.5 1.0 1.0\n")

    print("[preprocess] MNIST -> YOLO conversion complete.")


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 5,
) -> List[Path]:
    """Extract frames from a video file at the specified frame rate.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    output_dir:
        Directory where extracted frames are saved as PNGs.
    fps:
        Number of frames to capture per second of video.

    Returns
    -------
    list[Path]
        Paths to all extracted frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback assumption

    frame_interval = max(1, int(round(video_fps / fps)))
    stem = video_path.stem

    saved: List[Path] = []
    frame_idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"{stem}_frame_{save_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            save_idx += 1
        frame_idx += 1

    cap.release()
    print(
        f"[preprocess] Extracted {len(saved)} frames from {video_path.name} "
        f"(interval={frame_interval}, target_fps={fps})"
    )
    return saved


# ---------------------------------------------------------------------------
# Image normalisation
# ---------------------------------------------------------------------------

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE histogram equalization and min-max normalization.

    Parameters
    ----------
    image:
        Input BGR or grayscale image (uint8).

    Returns
    -------
    np.ndarray
        Normalised image as float32 in [0, 1].
    """
    if image.ndim == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    return image.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# YOLO label generation
# ---------------------------------------------------------------------------

def create_yolo_labels(
    annotations: List[Dict[str, Union[int, float, str]]],
    output_dir: str | Path,
) -> None:
    """Generate YOLO-format label ``.txt`` files from a list of annotations.

    Parameters
    ----------
    annotations:
        Each dict must contain:
        - ``image_name`` (str): filename without extension
        - ``class_id`` (int): zero-based class index
        - ``cx`` (float): normalised centre-x
        - ``cy`` (float): normalised centre-y
        - ``w`` (float): normalised width
        - ``h`` (float): normalised height
    output_dir:
        Directory to write ``.txt`` files into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image name
    grouped: Dict[str, List[Dict]] = {}
    for ann in annotations:
        name = ann["image_name"]
        grouped.setdefault(name, []).append(ann)

    for image_name, anns in grouped.items():
        label_path = output_dir / f"{image_name}.txt"
        with open(label_path, "w", encoding="utf-8") as fh:
            for a in anns:
                fh.write(
                    f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} "
                    f"{a['w']:.6f} {a['h']:.6f}\n"
                )

    print(f"[preprocess] Wrote {len(grouped)} YOLO label files to {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for ISL recognition."
    )
    sub = parser.add_subparsers(dest="command")

    # MNIST conversion
    mnist_p = sub.add_parser("convert-mnist", help="Convert MNIST CSV to YOLO format.")
    mnist_p.add_argument("--mnist-dir", required=True, help="Dir with MNIST CSVs.")
    mnist_p.add_argument("--output-dir", required=True, help="YOLO output root.")
    mnist_p.add_argument("--size", type=int, default=640, help="Target image size.")

    # Frame extraction
    frames_p = sub.add_parser("extract-frames", help="Extract video frames.")
    frames_p.add_argument("--video", required=True, help="Path to video file.")
    frames_p.add_argument("--output-dir", required=True, help="Output directory.")
    frames_p.add_argument("--fps", type=int, default=5, help="Frames per second.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "convert-mnist":
        convert_mnist_to_yolo(args.mnist_dir, args.output_dir, target_size=args.size)
    elif args.command == "extract-frames":
        extract_video_frames(args.video, args.output_dir, fps=args.fps)
    else:
        print("Usage: python -m data.preprocess {convert-mnist|extract-frames} ...")
