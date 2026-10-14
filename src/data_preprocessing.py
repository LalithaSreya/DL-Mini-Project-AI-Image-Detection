"""
Data preprocessing and dataset pipeline module for CIFAKE.
Provides dataset validation, image counting, and batch prefetching.
Conforms to Sreenidhi University Deep Learning Mini Project structural guidelines.
"""

from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    SUPPORTED_EXTENSIONS,
    DatasetNotFoundError,
    InvalidDatasetStructureError,
    validate_dataset_dir,
    count_dataset_images,
    load_datasets,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_LABEL",
    "LABEL_TO_CLASS",
    "SUPPORTED_EXTENSIONS",
    "DatasetNotFoundError",
    "InvalidDatasetStructureError",
    "validate_dataset_dir",
    "count_dataset_images",
    "load_datasets",
]
