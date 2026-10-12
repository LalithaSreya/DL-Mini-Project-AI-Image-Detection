# Test Suite Documentation

This directory contains automated unit and integration tests to verify pipeline integrity, preprocessing correctness, model definition, error handling, and evaluation computations.

## Purpose & Philosophy

To allow rapid development and verification without requiring the full 200 MB CIFAKE dataset or hours of GPU training:
- Tests utilize lightweight synthetic images (created in temporary directories) or mocked objects.
- Tests verify paths, class mappings, model layer shapes, metric calculations, and command-line arguments.
- Any student or grading faculty member can verify code correctness immediately in under 30 seconds.

## Running the Tests

To run the complete test suite with verbose output:

```bash
python -m pytest tests -v
```

To run a specific test:

```bash
python -m pytest tests/test_project.py -v
```

## Coverage of Test Cases

The test suite in `tests/test_project.py` verifies:

1. **Dataset Path Validation**:
   - Ensures an informative error is raised when `--data_dir` does not exist.
   - Ensures error handling detects missing subdirectories (such as `train` or `test`).
   - Ensures missing class directories (either `FAKE` or `REAL`) raise helpful descriptive exceptions.

2. **Image Counting and Dataset Loading**:
   - Validates that image files are discovered and counted correctly in a temporary dataset directory.
   - Verifies class mapping consistency (`FAKE = 0`, `REAL = 1`).

3. **Model Construction, Saving & Loading**:
   - Validates that the MobileNetV2 transfer learning model initializes with the expected input shape `(None, 160, 160, 3)`.
   - Confirms that the output is a single sigmoid activation tensor with shape `(None, 1)`.
   - Checks that the base MobileNetV2 backbone weights are frozen (`trainable = False`) as intended for initial transfer learning.
   - Verifies model serialization and deserialization (`.keras` format) without custom layer errors.

4. **Prediction Pipeline & Preprocessing**:
   - Tests `predict_image()` on a valid temporary image file.
   - Checks error handling on non-existent image paths.
   - Verifies score thresholds (score < 0.5 -> FAKE, score >= 0.5 -> REAL).

5. **Evaluation Metric Computation**:
   - Computes precision, recall, F1, accuracy, and confusion matrix against known ground-truth labels and predictions.
   - Confirms metrics JSON/CSV formatting and dictionary structures.

6. **Configuration Validation**:
   - Validates hyperparameter checking (e.g. rejecting negative epochs, invalid image sizes, or bad learning rates).
