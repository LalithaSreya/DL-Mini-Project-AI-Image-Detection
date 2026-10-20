"""
AI-Generated Image Detection Using Deep Learning
Package initialization for src modules.
"""

from src.dataset import validate_dataset_dir, count_dataset_images, load_datasets
from src.model import build_mobilenetv2_classifier
from src.train import run_training
from src.evaluate import evaluate_model, plot_and_save_history

__all__ = [
    "validate_dataset_dir",
    "count_dataset_images",
    "load_datasets",
    "build_mobilenetv2_classifier",
    "run_training",
    "evaluate_model",
    "plot_and_save_history",
]
