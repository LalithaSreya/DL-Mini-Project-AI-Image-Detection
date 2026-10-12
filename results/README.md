# Experimental Results and Evaluation Artifacts

This directory contains evaluation outputs, metric logs, and visualization figures produced from actual model training and testing runs.

## Academic Honesty & Experiment Status

> **STATUS: EXPERIMENTAL RUN PENDING REAL EXECUTION**
>
> In accordance with academic honesty principles, this repository does **NOT** contain pre-fabricated or simulated performance scores, fake confusion matrices, or copied benchmark tables. Metrics and visualization plots are strictly generated only when the training and evaluation pipelines are executed on an authentic CIFAKE dataset split.

## Expected Generated Files

When `main.py` (or `src/evaluate.py`) is run with `--evaluate_test`, the following artifacts will be automatically saved to this directory:

| Filename | Description |
| :--- | :--- |
| `metrics.json` | Detailed numerical test results (Test Loss, Accuracy, Precision, Recall, F1-Score, Per-Class Metrics, Support counts). |
| `metrics.csv` | Tabular summary of overall and per-class performance metrics. |
| `training_loss_curve.png` | Plot comparing Training Loss vs. Validation Loss across all completed epochs. |
| `training_accuracy_curve.png` | Plot comparing Training Accuracy vs. Validation Accuracy across epochs. |
| `confusion_matrix.png` | Seaborn/Matplotlib annotated heatmap showing True vs. Predicted counts for `FAKE` and `REAL` classes. |

## Class Label Mapping & Evaluation Conventions

Across all evaluation routines, the class mapping is strictly defined as:

| Class Name | Integer Label | Description |
| :--- | :---: | :--- |
| **FAKE** | `0` | AI-generated synthetic images (Stable Diffusion v1.4) |
| **REAL** | `1` | Authentic photographic images (CIFAR-10) |

### Confusion Matrix Matrix Orientation:
- **Rows**: Ground Truth (`0: FAKE`, `1: REAL`)
- **Columns**: Model Predictions (`0: FAKE`, `1: REAL`)
  - Row 0, Col 0: True Negatives (Correctly identified FAKE)
  - Row 0, Col 1: False Positives (FAKE incorrectly classified as REAL)
  - Row 1, Col 0: False Negatives (REAL incorrectly classified as FAKE)
  - Row 1, Col 1: True Positives (Correctly identified REAL)

### Metric Definitions:
- **Per-Class Precision, Recall, and F1-Score**: Reported individually for both `FAKE` (class 0) and `REAL` (class 1).
- **Macro-Averaged F1-Score**: Unweighted mean of F1 scores across both classes, ensuring balanced evaluation regardless of sample distributions.
- **Accuracy**: Overall fraction of correct classifications across the test split.

## How to Populate Results
To execute the pipeline and generate real results:
```bash
python main.py --data_dir path/to/CIFAKE --epochs 10 --batch_size 32
```
