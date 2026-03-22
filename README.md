# Bridging Silent Worlds: Advanced Computer Vision Approaches to Indian Sign Language Recognition

A deep-learning pipeline for real-time **Indian Sign Language (ISL)** gesture recognition, combining YOLOv5/v8 object detection, ensemble inference, emotion-aware context, and bidirectional gesture-text translation.

---

## Highlights

- **Dual detector architecture** -- YOLOv5 and YOLOv8 models with an ensemble combiner for robust detection
- **Real-time webcam inference** -- sustained-detection logic filters noise and produces stable predictions
- **Bidirectional translation** -- gesture-to-text and text-to-gesture conversion
- **Emotion-aware pipeline** -- sentiment analysis adds emotional context to translated text
- **Multi-dataset support** -- Sign Language MNIST, custom ISL gestures, and video-based CSLTR sequences
- **Augmentation pipeline** -- Albumentations transforms designed for sign language (no horizontal flip)

## Project Structure

```
.
├── config/                 # Settings, YOLOv5/v8 training configs
│   ├── settings.py         # Centralized dataclass-based configuration
│   ├── yolov5_config.yaml
│   └── yolov8_config.yaml
├── data/                   # Data pipeline
│   ├── download.py         # Dataset download & directory scaffolding
│   ├── preprocess.py       # Resize, MNIST-to-YOLO conversion, frame extraction
│   ├── augment.py          # Training/validation transform pipelines
│   ├── dataset.py          # PyTorch Dataset classes (image, MNIST, video, combined)
│   └── split.py            # Stratified train/val/test splitting
├── models/                 # Model definitions
│   ├── yolov5_detector.py  # YOLOv5-based gesture detector
│   ├── yolov8_detector.py  # YOLOv8-based gesture detector
│   ├── ensemble.py         # Weighted ensemble of multiple detectors
│   ├── emotion_classifier.py  # DistilBERT sentiment classifier
│   └── gesture_vocabulary.py  # Gesture class mappings & ISL alphabet
├── training/               # Training & evaluation
│   ├── train_yolov5.py     # YOLOv5 training script
│   ├── train_yolov8.py     # YOLOv8 training script
│   ├── train_emotion.py    # Emotion classifier fine-tuning
│   ├── evaluate.py         # Model evaluation & metrics
│   └── callbacks.py        # Training callbacks (logging, checkpoints, early stop)
├── inference/              # Inference & translation
│   ├── detector.py         # Unified detection interface
│   ├── webcam.py           # Real-time webcam detection loop
│   ├── gesture_to_text.py  # Gesture sequence to text translation
│   ├── text_to_gesture.py  # Text to gesture sequence mapping
│   └── emotion_pipeline.py # End-to-end emotion-aware inference
├── visualization/          # Plotting & exploration
│   ├── dataset_explorer.py # Interactive dataset browsing
│   ├── detection_overlay.py# Bounding box & label rendering
│   └── metrics.py          # Confusion matrix, PR curves, loss plots
├── Makefile                # Common commands (train, detect, test, etc.)
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
└── LICENSE                 # MIT License
```

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- Kaggle API token (for MNIST dataset download)

### Installation

```bash
# Clone the repository
git clone https://github.com/therealpratyushraman/Bridging-Silent-Worlds-Advanced-Computer-Vision-Approaches-to-Indian-Sign-Language-Recognition.git
cd Bridging-Silent-Worlds-Advanced-Computer-Vision-Approaches-to-Indian-Sign-Language-Recognition

# Install dependencies
make install
# or manually:
pip install -r requirements.txt
pip install -e .
```

### Download Data

```bash
# Sign Language MNIST (requires ~/.kaggle/kaggle.json)
python -m data.download --dataset mnist

# Custom ISL gesture dataset scaffold
python -m data.download --dataset custom

# Video-based CSLTR dataset scaffold
python -m data.download --dataset csltr
```

### Preprocess & Split

```bash
# Convert MNIST CSV to YOLO image/label format
python -m data.preprocess convert-mnist

# Stratified train/val/test split
python -m data.split --data-dir data/raw/isl_custom/raw --output-dir data/raw/isl_custom

# Augment the training set
python -m data.augment --data-dir data/raw/isl_custom/images/train --multiplier 3
```

## Training

```bash
# Train YOLOv8 detector (default)
make train-yolov8

# Train YOLOv5 detector
make train-yolov5

# Train emotion classifier
make train-emotion

# Evaluate trained models
make evaluate
```

Training parameters are configured in `config/yolov8_config.yaml` and `config/settings.py`. Key defaults:

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | SGD (momentum 0.937) |
| Learning rate | 0.01 |
| Warmup epochs | 3 |

## Inference

### Real-Time Webcam Detection

```bash
# Launch webcam detection (camera index 0)
make detect
# or:
python -m inference.webcam --source 0
```

The webcam pipeline uses **sustained detection** (default 3 seconds) to filter transient misclassifications before committing to a prediction.

### Programmatic Usage

```python
from models import YOLOv8Detector, EnsembleDetector, EmotionClassifier
from inference import GestureToTextConverter, SignLanguageDetector

# Single detector
detector = YOLOv8Detector(weights="runs/train/best.pt")
results = detector.predict("path/to/image.jpg")

# Ensemble of YOLOv5 + YOLOv8
ensemble = EnsembleDetector(detectors=[yolov5, yolov8], weights=[0.4, 0.6])
results = ensemble.predict(frame)

# Gesture to text
converter = GestureToTextConverter()
text = converter.convert(gesture_sequence)

# Emotion-aware pipeline
classifier = EmotionClassifier()
sentiment = classifier.predict(text)
```

## Datasets

| Dataset | Type | Classes | Size |
|---------|------|---------|------|
| Sign Language MNIST | Static images (CSV) | 24 letters (A-Y, excl. J, Z) | 34,627 |
| Custom ISL Gestures | Static images | 6 gestures (Hello, Help, Home, No, Please, Yes) | ~2,000 |
| ISL CSLTR | Video sequences | Continuous signing | Variable |

See [`data/README.md`](data/README.md) for detailed dataset documentation.

## Augmentation Strategy

Training augmentations are applied via [Albumentations](https://albumentations.ai/):

- **RandomBrightnessContrast** -- varied lighting simulation
- **GaussNoise** -- sensor noise robustness
- **MotionBlur** -- real-time motion tolerance
- **ShiftScaleRotate** (up to 15 deg) -- viewpoint variation

**HorizontalFlip is intentionally excluded** because mirroring changes the meaning of hand-specific sign language gestures.

## Visualization

```bash
# Explore dataset samples
python -m visualization.dataset_explorer

# Plot training metrics (confusion matrix, PR curves, loss)
python -m visualization.metrics
```

## Testing

```bash
make test
# Runs: pytest tests/ -v --cov=. --cov-report=term-missing
```

## Configuration

All settings are centralized in `config/settings.py` using Python dataclasses:

```python
from config import get_config

cfg = get_config()
print(cfg.model.gesture_classes)   # ['Hello', 'Help', 'Home', 'No', 'Please', 'Yes']
print(cfg.training.epochs)          # 100
print(cfg.device)                   # 'cuda' or 'cpu'
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Pratyush Raman**

---

*Built with PyTorch, Ultralytics, OpenCV, MediaPipe, and Hugging Face Transformers.*
