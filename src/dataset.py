"""
Dataset loading, verification, and preprocessing for CIFAKE.
Binary Classes:
    0: FAKE (AI-generated synthetic images)
    1: REAL (Authentic natural images)
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import tensorflow as tf

# Canonical class mappings across all modules
CLASS_NAMES = ["FAKE", "REAL"]
CLASS_TO_LABEL = {"FAKE": 0, "REAL": 1}
LABEL_TO_CLASS = {0: "FAKE", 1: "REAL"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset directory or expected subdirectories are not found."""
    pass


class InvalidDatasetStructureError(ValueError):
    """Raised when required class folders or images are missing."""
    pass


def validate_dataset_dir(data_dir: str | Path) -> Dict[str, Path]:
    """
    Validates that the provided dataset path contains the required splits and class folders.

    Expected structure:
        data_dir/
            train/
                FAKE/
                REAL/
            test/
                FAKE/
                REAL/

    Returns:
        dict: Paths to train_dir, test_dir, and individual class directories.

    Raises:
        DatasetNotFoundError: If data_dir or split directories are missing.
        InvalidDatasetStructureError: If class folders are missing.
    """
    path = Path(data_dir)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset directory does not exist: '{path.resolve()}'.\n"
            "Please download CIFAKE from Kaggle and verify your path."
        )

    train_dir = path / "train"
    test_dir = path / "test"

    missing_splits = []
    if not train_dir.is_dir():
        missing_splits.append("train")
    if not test_dir.is_dir():
        missing_splits.append("test")

    if missing_splits:
        raise DatasetNotFoundError(
            f"Missing required split directories in '{path.resolve()}': {missing_splits}.\n"
            "Expected 'train/' and 'test/' subdirectories inside the dataset directory."
        )

    required_classes = ["FAKE", "REAL"]
    missing_folders = []

    for split_name, split_path in [("train", train_dir), ("test", test_dir)]:
        for c in required_classes:
            class_folder = split_path / c
            if not class_folder.is_dir():
                missing_folders.append(f"{split_name}/{c}")

    if missing_folders:
        raise InvalidDatasetStructureError(
            f"Missing class directories in '{path.resolve()}': {missing_folders}.\n"
            "Both 'FAKE' and 'REAL' class folders must exist under each split."
        )

    return {
        "root": path,
        "train": train_dir,
        "test": test_dir,
        "train_fake": train_dir / "FAKE",
        "train_real": train_dir / "REAL",
        "test_fake": test_dir / "FAKE",
        "test_real": test_dir / "REAL",
    }


def count_dataset_images(data_dir: str | Path) -> Dict[str, Dict[str, int]]:
    """
    Scans and counts valid images in each split and class.

    Returns:
        dict: Counts of images per split and class, e.g.:
              {'train': {'FAKE': 50000, 'REAL': 50000, 'total': 100000}, ...}
    """
    paths = validate_dataset_dir(data_dir)
    counts = {}

    for split in ["train", "test"]:
        split_counts = {}
        total = 0
        for c in CLASS_NAMES:
            folder = paths[f"{split}_{c.lower()}"]
            n = sum(
                1 for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            split_counts[c] = n
            total += n
        split_counts["total"] = total
        counts[split] = split_counts

    return counts


def load_datasets(
    data_dir: str | Path,
    image_size: Tuple[int, int] = (160, 160),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads training, validation, and test datasets.
    - Training data is partitioned into train and validation splits.
    - Test data is strictly kept separate and held out.
    - Explicit class_names=['FAKE', 'REAL'] enforces:
        FAKE -> 0
        REAL -> 1

    Args:
        data_dir: Root dataset folder containing train/ and test/.
        image_size: Target height and width (H, W).
        batch_size: Batch size for DataLoader.
        validation_split: Fraction of training data reserved for validation.
        seed: Random seed for reproducible splitting and shuffling.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    paths = validate_dataset_dir(data_dir)

    # 1. Training split (from train/ folder)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(paths["train"]),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    # 2. Validation split (from train/ folder)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(paths["train"]),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Held-out test split (from test/ folder, never used for training)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(paths["test"]),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Performance optimization with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
