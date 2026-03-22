"""PyTorch Dataset classes for Indian Sign Language recognition.

Provides datasets for:
- YOLO-format image detection data (``ISLImageDataset``)
- Sign Language MNIST CSV files (``ISLMNISTDataset``)
- Video frame sequences (``ISLVideoDataset``)
- Combined multi-source datasets (``CombinedISLDataset``)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# YOLO-format image dataset
# ---------------------------------------------------------------------------

class ISLImageDataset(Dataset):
    """Dataset for YOLO-format image detection data.

    Expects a directory layout of::

        root_dir/
            images/<split>/   *.png / *.jpg
            labels/<split>/   *.txt  (YOLO format: class cx cy w h)

    Parameters
    ----------
    root_dir:
        Root of the YOLO-format dataset.
    transform:
        Optional callable that takes a numpy BGR image and returns a
        transformed image (numpy array or torch Tensor).
    split:
        One of ``'train'``, ``'val'``, or ``'test'``.
    """

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        split: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        self.image_dir = self.root_dir / "images" / split
        self.label_dir = self.root_dir / "labels" / split

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        self.image_paths: List[Path] = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self._IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return a sample dict with keys ``image``, ``labels``, ``image_path``.

        ``labels`` is a float tensor of shape ``(N, 5)`` where each row is
        ``[class_id, cx, cy, w, h]`` in normalised coordinates.  If no
        label file exists, an empty ``(0, 5)`` tensor is returned.
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")

        # Convert BGR -> RGB for consistency with most transforms
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load YOLO labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path)

        if self.transform is not None:
            image = self.transform(image)

        # Ensure image is a tensor
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = torch.from_numpy(image)
            else:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
            # HWC -> CHW
            if image.ndim == 3 and image.shape[2] in (1, 3):
                image = image.permute(2, 0, 1)

        return {
            "image": image,
            "labels": labels,
            "image_path": str(img_path),
        }

    @staticmethod
    def _load_labels(label_path: Path) -> torch.Tensor:
        """Parse a YOLO label file into a float tensor."""
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        rows: List[List[float]] = []
        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    rows.append([float(x) for x in parts])

        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sign Language MNIST CSV dataset
# ---------------------------------------------------------------------------

class ISLMNISTDataset(Dataset):
    """Dataset wrapper for Sign Language MNIST CSV files.

    The CSV format has a header row followed by rows where column 0 is
    the label (0-24, mapping to A-Y excluding J and Z) and columns 1-784
    are grayscale pixel values for a 28x28 image.

    Parameters
    ----------
    csv_path:
        Path to the ``sign_mnist_train.csv`` or ``sign_mnist_test.csv``.
    transform:
        Optional callable applied to the 28x28 uint8 numpy image.
    """

    # Sign Language MNIST label -> letter mapping (J=9 and Z=25 excluded)
    _LABEL_TO_LETTER: Dict[int, str] = {
        i: chr(ord("A") + i + (1 if i >= 9 else 0))
        for i in range(25)
        if i != 9  # label 9 would be J, which is excluded
    }

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.transform = transform

        self.labels: List[int] = []
        self.pixels: List[np.ndarray] = []

        with open(self.csv_path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                self.labels.append(int(row[0]))
                self.pixels.append(
                    np.array(row[1:], dtype=np.uint8).reshape(28, 28)
                )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return dict with ``image``, ``label`` (int), ``letter`` (str)."""
        image = self.pixels[idx].copy()
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        # Default: convert to float tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale -> single-channel tensor
                image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
                if image.ndim == 3 and image.shape[2] in (1, 3):
                    image = image.permute(2, 0, 1)

        letter = self._LABEL_TO_LETTER.get(label, "?")
        return {
            "image": image,
            "label": label,
            "letter": letter,
        }

    def get_class_distribution(self) -> Dict[int, int]:
        """Return a mapping of label -> count for the full dataset."""
        dist: Dict[int, int] = {}
        for lbl in self.labels:
            dist[lbl] = dist.get(lbl, 0) + 1
        return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# Video frame sequence dataset
# ---------------------------------------------------------------------------

class ISLVideoDataset(Dataset):
    """Dataset for video frame sequences (continuous signing).

    Loads fixed-length clips of consecutive frames from per-video
    subdirectories.  Designed for temporal models (3D-CNN, LSTM, etc.).

    Expected layout::

        root_dir/
            <split>/
                <video_id>/
                    frame_000000.png
                    frame_000001.png
                    ...

    Parameters
    ----------
    root_dir:
        Root frames directory (e.g. ``isl_csltr/frames``).
    transform:
        Optional callable applied to each individual frame (numpy BGR).
    clip_length:
        Number of consecutive frames per sample clip.
    split:
        One of ``'train'``, ``'val'``, or ``'test'``.
    """

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        clip_length: int = 16,
        split: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.clip_length = clip_length
        self.split = split

        self.clips: List[Tuple[List[Path], str]] = []
        self._build_clip_index()

    def _build_clip_index(self) -> None:
        """Index all valid clips from the frame directories."""
        if not self.root_dir.is_dir():
            return

        for video_dir in sorted(self.root_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            frames = sorted(
                p for p in video_dir.iterdir()
                if p.suffix.lower() in self._IMAGE_EXTENSIONS
            )

            video_id = video_dir.name

            # Slide a window of clip_length over available frames
            if len(frames) < self.clip_length:
                # Pad short videos by repeating the last frame
                if frames:
                    padded = frames + [frames[-1]] * (self.clip_length - len(frames))
                    self.clips.append((padded, video_id))
            else:
                # Non-overlapping clips
                for start in range(0, len(frames) - self.clip_length + 1, self.clip_length):
                    clip_frames = frames[start : start + self.clip_length]
                    self.clips.append((clip_frames, video_id))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return dict with ``frames`` (T, C, H, W tensor), ``video_id``."""
        frame_paths, video_id = self.clips[idx]

        frames: List[torch.Tensor] = []
        for fp in frame_paths:
            image = cv2.imread(str(fp))
            if image is None:
                raise IOError(f"Failed to read frame: {fp}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                image = self.transform(image)

            if isinstance(image, np.ndarray):
                if image.dtype != np.float32:
                    image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image)
                if image.ndim == 3 and image.shape[2] in (1, 3):
                    image = image.permute(2, 0, 1)

            frames.append(image)

        # Stack into (T, C, H, W)
        clip_tensor = torch.stack(frames, dim=0)
        return {
            "frames": clip_tensor,
            "video_id": video_id,
        }


# ---------------------------------------------------------------------------
# Combined multi-source dataset
# ---------------------------------------------------------------------------

class CombinedISLDataset(Dataset):
    """Merges multiple ISL datasets with a consistent label mapping.

    Each constituent dataset must return dicts containing at least a
    ``label`` or ``labels`` key.  This wrapper applies a global label
    remapping so that class indices are consistent across all sources.

    Parameters
    ----------
    datasets:
        Sequence of ``Dataset`` instances to combine.
    label_map:
        Optional global label mapping.  Keys are ``(dataset_index,
        original_label)`` tuples, values are the unified label integer.
        When ``None``, labels are passed through unchanged.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        label_map: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> None:
        self.datasets = list(datasets)
        self.label_map = label_map

        # Build cumulative length index for O(1) lookup
        self._cumulative_lengths: List[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cumulative_lengths.append(total)

    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return sample from the appropriate sub-dataset with remapped labels."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for combined dataset of size {len(self)}")

        # Determine which dataset and local index
        ds_idx = 0
        offset = 0
        for i, cum_len in enumerate(self._cumulative_lengths):
            if idx < cum_len:
                ds_idx = i
                break
            offset = cum_len

        local_idx = idx - offset
        sample = self.datasets[ds_idx][local_idx]

        # Remap label if mapping is provided
        if self.label_map is not None:
            if "label" in sample:
                original = sample["label"]
                key = (ds_idx, int(original))
                if key in self.label_map:
                    sample["label"] = self.label_map[key]

        # Tag with source dataset index
        sample["source_dataset"] = ds_idx
        return sample
