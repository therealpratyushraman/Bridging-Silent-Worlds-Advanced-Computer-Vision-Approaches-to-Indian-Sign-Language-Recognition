# Data Module

Dataset download, preprocessing, augmentation, and loading utilities for
the Indian Sign Language (ISL) recognition system.

## Supported Datasets

### 1. Sign Language MNIST (Kaggle)

- **Source:** <https://www.kaggle.com/datasets/datamunge/sign-language-mnist>
- **Size:** 34,627 images (27,455 train / 7,172 test)
- **Classes:** 24 hand-sign letters (A-Y, excluding J and Z which require motion)
- **Format:** 28x28 grayscale pixel arrays stored in CSV files
- **Labels:** Integer 0-24 mapping to the 24 static alphabet letters

Download requires a Kaggle account and API token:

```bash
pip install kaggle
# Place your token at ~/.kaggle/kaggle.json
python -m data.download --dataset mnist
```

### 2. Custom ISL Gesture Dataset

- **Size:** 2,000 images (target)
- **Classes:** 6 gestures — Hello, Help, Home, No, Please, Yes
- **Collection:** Manual capture with varied backgrounds, lighting, and hand positions

Set up the directory structure, then populate with collected images:

```bash
python -m data.download --dataset custom
# Place images in data/raw/isl_custom/raw/<ClassName>/
python -m data.preprocess convert-mnist  # (if converting from a flat format)
python -m data.split --data-dir data/raw/isl_custom/raw --output-dir data/raw/isl_custom
```

### 3. ISL CSLTR (Continuous Sign Language Translation and Recognition)

- **Type:** Video-based continuous signing sequences
- **Format:** Video files (MP4/AVI) with per-frame or segment annotations
- **Use case:** Temporal models (3D-CNN, LSTM, Transformer)

Set up and extract frames:

```bash
python -m data.download --dataset csltr
# Place videos in data/raw/isl_csltr/videos/
python -m data.preprocess extract-frames --video data/raw/isl_csltr/videos/example.mp4 --output-dir data/raw/isl_csltr/frames/train/example
```

## Expected Directory Structure

After setup and preprocessing, the YOLO-format layout is:

```
data/
├── raw/
│   ├── sign_language_mnist/
│   │   ├── sign_mnist_train.csv
│   │   ├── sign_mnist_test.csv
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── dataset.yaml
│   ├── isl_custom/
│   │   ├── raw/
│   │   │   ├── Hello/
│   │   │   ├── Help/
│   │   │   ├── Home/
│   │   │   ├── No/
│   │   │   ├── Please/
│   │   │   └── Yes/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── dataset.yaml
│   └── isl_csltr/
│       ├── videos/
│       ├── annotations/
│       ├── frames/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── __init__.py
├── download.py
├── preprocess.py
├── augment.py
├── dataset.py
├── split.py
└── README.md
```

## Module Components

| File | Purpose |
|------|---------|
| `download.py` | Dataset download (Kaggle API) and directory scaffolding |
| `preprocess.py` | Letterbox resize, MNIST-to-YOLO conversion, video frame extraction, normalisation, YOLO label generation |
| `augment.py` | Albumentations-based training/validation transform pipelines and batch augmentation |
| `dataset.py` | PyTorch `Dataset` classes for images, MNIST CSV, video clips, and combined multi-source loading |
| `split.py` | Stratified train/val/test splitting with class-balance preservation |

## Quick Start

```python
from data import (
    download_sign_language_mnist,
    convert_mnist_to_yolo,
    get_train_transforms,
    ISLImageDataset,
    stratified_split,
)

# 1. Download
mnist_dir = download_sign_language_mnist("data/raw")

# 2. Convert to YOLO format
convert_mnist_to_yolo(mnist_dir, mnist_dir, target_size=640)

# 3. Create PyTorch dataset with augmentation
transform = get_train_transforms(image_size=640)
dataset = ISLImageDataset(root_dir=mnist_dir, transform=transform, split="train")

# 4. Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
```

## Augmentation Strategy

Training augmentations include:
- **RandomBrightnessContrast** -- simulates varied lighting
- **GaussNoise** -- sensor noise robustness
- **MotionBlur** -- real-time capture motion tolerance
- **ShiftScaleRotate** (up to 15 degrees) -- viewpoint variation

**HorizontalFlip is intentionally excluded** because mirroring changes the
meaning of hand-specific sign language gestures.
