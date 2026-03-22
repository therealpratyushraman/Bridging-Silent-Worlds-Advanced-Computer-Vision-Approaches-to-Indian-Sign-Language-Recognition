"""Data module for Indian Sign Language Recognition system.

Provides dataset download, preprocessing, augmentation, PyTorch dataset
classes, and train/val/test splitting utilities.
"""

from data.download import (
    download_sign_language_mnist,
    setup_custom_dataset,
    setup_isl_csltr,
    generate_dataset_yaml,
)
from data.preprocess import (
    letterbox_resize,
    convert_mnist_to_yolo,
    extract_video_frames,
    normalize_image,
    create_yolo_labels,
)
from data.augment import (
    get_train_transforms,
    get_val_transforms,
    augment_dataset,
    augment_single_image,
)
from data.dataset import (
    ISLImageDataset,
    ISLMNISTDataset,
    ISLVideoDataset,
    CombinedISLDataset,
)
from data.split import (
    stratified_split,
    copy_split_files,
)

__all__ = [
    # download
    "download_sign_language_mnist",
    "setup_custom_dataset",
    "setup_isl_csltr",
    "generate_dataset_yaml",
    # preprocess
    "letterbox_resize",
    "convert_mnist_to_yolo",
    "extract_video_frames",
    "normalize_image",
    "create_yolo_labels",
    # augment
    "get_train_transforms",
    "get_val_transforms",
    "augment_dataset",
    "augment_single_image",
    # dataset
    "ISLImageDataset",
    "ISLMNISTDataset",
    "ISLVideoDataset",
    "CombinedISLDataset",
    # split
    "stratified_split",
    "copy_split_files",
]
