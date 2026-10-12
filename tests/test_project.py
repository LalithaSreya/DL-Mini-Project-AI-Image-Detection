"""
Automated unit and integration test suite for AI-Generated Image Detection.
Covers:
1. Dataset path validation & missing directory errors
2. Class mapping consistency
3. Lightweight synthetic dataset verification
4. Model architecture construction & output tensor shapes
5. Prediction input validation & error handling
6. Metric calculations (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import os
import json
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

from src.dataset import (
    validate_dataset_dir,
    count_dataset_images,
    DatasetNotFoundError,
    InvalidDatasetStructureError,
    CLASS_NAMES,
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
)
from src.model import build_mobilenetv2_classifier
from predict import load_and_preprocess_image, predict_image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def create_dummy_image(file_path: Path, size=(32, 32), color=(255, 0, 0)):
    """Helper to generate a lightweight synthetic test image."""
    img = Image.new("RGB", size, color=color)
    img.save(file_path)


# ---------------------------------------------------------------------------
# 1. Dataset Path Validation & Error Handling Tests
# ---------------------------------------------------------------------------
def test_dataset_path_nonexistent(tmp_path):
    """Test that validating a non-existent path raises DatasetNotFoundError."""
    non_existent = tmp_path / "does_not_exist_dir"
    with pytest.raises(DatasetNotFoundError) as exc_info:
        validate_dataset_dir(non_existent)
    assert "does not exist" in str(exc_info.value)


def test_dataset_missing_splits(tmp_path):
    """Test that missing train/ or test/ subdirectories raises DatasetNotFoundError."""
    dataset_root = tmp_path / "incomplete_dataset"
    dataset_root.mkdir()
    # Create train but leave out test
    (dataset_root / "train").mkdir()
    with pytest.raises(DatasetNotFoundError) as exc_info:
        validate_dataset_dir(dataset_root)
    assert "Missing required split directories" in str(exc_info.value)


def test_dataset_missing_class_folders(tmp_path):
    """Test that missing FAKE or REAL class folders raises InvalidDatasetStructureError."""
    dataset_root = tmp_path / "partial_classes"
    (dataset_root / "train" / "FAKE").mkdir(parents=True)
    # Missing train/REAL
    (dataset_root / "test" / "FAKE").mkdir(parents=True)
    (dataset_root / "test" / "REAL").mkdir(parents=True)

    with pytest.raises(InvalidDatasetStructureError) as exc_info:
        validate_dataset_dir(dataset_root)
    assert "Missing class directories" in str(exc_info.value)
    assert "train/REAL" in str(exc_info.value)


def test_dataset_valid_structure_and_counting(tmp_path):
    """Test that a valid directory structure is recognized and images are counted correctly."""
    dataset_root = tmp_path / "mini_cifake"
    for split in ["train", "test"]:
        for c in ["FAKE", "REAL"]:
            folder = dataset_root / split / c
            folder.mkdir(parents=True)
            # Create 2 dummy images in each folder
            create_dummy_image(folder / "img_1.jpg")
            create_dummy_image(folder / "img_2.png")

    paths = validate_dataset_dir(dataset_root)
    assert paths["train"].exists()
    assert paths["test"].exists()

    counts = count_dataset_images(dataset_root)
    assert counts["train"]["FAKE"] == 2
    assert counts["train"]["REAL"] == 2
    assert counts["train"]["total"] == 4
    assert counts["test"]["FAKE"] == 2
    assert counts["test"]["REAL"] == 2
    assert counts["test"]["total"] == 4


# ---------------------------------------------------------------------------
# 2. Canonical Class Mapping Tests
# ---------------------------------------------------------------------------
def test_class_mappings():
    """Ensure strict, unambiguous class mapping across the entire system."""
    assert CLASS_NAMES == ["FAKE", "REAL"]
    assert CLASS_TO_LABEL["FAKE"] == 0
    assert CLASS_TO_LABEL["REAL"] == 1
    assert LABEL_TO_CLASS[0] == "FAKE"
    assert LABEL_TO_CLASS[1] == "REAL"


# ---------------------------------------------------------------------------
# 3. Model Architecture Construction Tests
# ---------------------------------------------------------------------------
def test_model_construction():
    """
    Test building the MobileNetV2 classifier.
    Uses weights=None to test architecture instantiation swiftly without downloading ImageNet weights.
    """
    input_shape = (160, 160, 3)
    model = build_mobilenetv2_classifier(
        input_shape=input_shape,
        learning_rate=1e-4,
        dropout_rate=0.2,
        weights=None,
        freeze_backbone=True,
    )

    # Validate model input and output shapes
    assert model.input_shape == (None, 160, 160, 3)
    assert model.output_shape == (None, 1)

    # Validate output activation layer
    last_layer = model.layers[-1]
    assert isinstance(last_layer, tf.keras.layers.Dense)
    assert last_layer.activation.__name__ == "sigmoid"

    # Validate optimizer and loss
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert isinstance(model.loss, tf.keras.losses.BinaryCrossentropy)


def test_model_save_and_load(tmp_path):
    """
    Test that the constructed model can be saved to disk and loaded back seamlessly
    without custom layer deserialization errors.
    """
    model = build_mobilenetv2_classifier(
        input_shape=(160, 160, 3),
        weights=None,
    )
    checkpoint_file = tmp_path / "test_detector.keras"
    model.save(str(checkpoint_file))
    assert checkpoint_file.exists()

    loaded_model = tf.keras.models.load_model(str(checkpoint_file))
    assert loaded_model.input_shape == (None, 160, 160, 3)
    assert loaded_model.output_shape == (None, 1)

    # Verify predictions on a synthetic input match
    dummy_input = np.ones((1, 160, 160, 3), dtype=np.float32) * 128.0
    pred_orig = model.predict(dummy_input, verbose=0)
    pred_loaded = loaded_model.predict(dummy_input, verbose=0)
    np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Prediction Input Validation & Processing Tests
# ---------------------------------------------------------------------------
def test_load_and_preprocess_image_valid(tmp_path):
    """Test loading and preprocessing a valid image file."""
    img_path = tmp_path / "valid_sample.jpg"
    create_dummy_image(img_path, size=(64, 64))

    tensor = load_and_preprocess_image(img_path, target_size=(160, 160))
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 160, 160, 3)
    # Check pixel value range (expected [0, 255] before model internal layer)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 255.0


def test_load_and_preprocess_image_missing(tmp_path):
    """Test that missing image file raises FileNotFoundError."""
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_image(missing)


def test_load_and_preprocess_image_corrupted(tmp_path):
    """Test that corrupted non-image file raises ValueError."""
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_text("This is plain text, not a JPEG image.")

    with pytest.raises(ValueError):
        load_and_preprocess_image(bad_file)


def test_predict_image_thresholding(tmp_path):
    """Test that sigmoid probability thresholding correctly outputs REAL vs FAKE."""
    img_path = tmp_path / "test_sample.png"
    create_dummy_image(img_path, size=(32, 32))

    # Mock model predicting REAL (prob = 0.85)
    class MockModelReal:
        def predict(self, x, verbose=0):
            return np.array([[0.85]], dtype=np.float32)

    pred_class, raw_score, conf = predict_image(img_path, MockModelReal())
    assert pred_class == "REAL"
    assert raw_score == pytest.approx(0.85, abs=1e-4)
    assert conf == pytest.approx(85.0, abs=0.1)

    # Mock model predicting FAKE (prob = 0.15)
    class MockModelFake:
        def predict(self, x, verbose=0):
            return np.array([[0.15]], dtype=np.float32)

    pred_class, raw_score, conf = predict_image(img_path, MockModelFake())
    assert pred_class == "FAKE"
    assert raw_score == pytest.approx(0.15, abs=1e-4)
    assert conf == pytest.approx(85.0, abs=0.1)


# ---------------------------------------------------------------------------
# 5. Metric Calculation Tests
# ---------------------------------------------------------------------------
def test_metric_calculations():
    """Verify precision, recall, F1, and confusion matrix computations on known arrays."""
    # Ground truth: 3 FAKE (0), 3 REAL (1)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Predicted: 2 correctly FAKE, 1 FAKE misclassified as REAL, 3 correctly REAL
    y_pred = np.array([0, 0, 1, 1, 1, 1])

    acc = accuracy_score(y_true, y_pred)
    assert acc == pytest.approx(5 / 6, abs=1e-4)

    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1]
    )
    # FAKE (0): TP=2, FP=0, FN=1 -> Precision=2/2=1.0, Recall=2/3=0.6667
    assert p_per[0] == 1.0
    assert r_per[0] == pytest.approx(2 / 3, abs=1e-4)
    # REAL (1): TP=3, FP=1, FN=0 -> Precision=3/4=0.75, Recall=3/3=1.0
    assert p_per[1] == 0.75
    assert r_per[1] == 1.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm[0, 0] == 2  # True Negative (FAKE)
    assert cm[0, 1] == 1  # False Positive (classified as REAL)
    assert cm[1, 0] == 0  # False Negative (classified as FAKE)
    assert cm[1, 1] == 3  # True Positive (REAL)
