"""
Evaluation pipeline for MobileNetV2 on the CIFAKE held-out test split.
Calculates:
- Loss, Accuracy, Precision, Recall, F1-score (overall, macro-average, and per-class)
- Confusion Matrix (Ground Truth Rows x Predicted Columns: 0=FAKE, 1=REAL)
- Plots: Loss Curves, Accuracy Curves, and Confusion Matrix Heatmap
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import tensorflow as tf

from src.dataset import load_datasets, CLASS_NAMES


def plot_and_save_history(
    history: Dict[str, list],
    results_dir: str | Path = "results",
) -> Tuple[Path, Path]:
    """
    Generates and saves training vs. validation loss and accuracy curves.

    Args:
        history: Dictionary containing keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        results_dir: Directory where PNG plots will be stored.

    Returns:
        (loss_plot_path, accuracy_plot_path)
    """
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("loss", [])) + 1)

    # 1. Training vs Validation Loss Curve
    loss_path = res_path / "training_loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("loss", []), "b-o", label="Training Loss")
    if "val_loss" in history:
        plt.plot(epochs, history.get("val_loss", []), "r--s", label="Validation Loss")
    plt.title("Model Loss vs. Epochs (MobileNetV2)")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2. Training vs Validation Accuracy Curve
    acc_path = res_path / "training_accuracy_curve.png"
    plt.figure(figsize=(8, 5))
    acc_key = "accuracy" if "accuracy" in history else "binary_accuracy"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_binary_accuracy"
    plt.plot(epochs, history.get(acc_key, []), "b-o", label="Training Accuracy")
    if val_acc_key in history:
        plt.plot(epochs, history.get(val_acc_key, []), "g--^", label="Validation Accuracy")
    plt.title("Model Accuracy vs. Epochs (MobileNetV2)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    return loss_path, acc_path


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    results_dir: str | Path = "results",
) -> Path:
    """
    Plots an annotated confusion matrix heatmap and saves to results/.
    Class 0: FAKE, Class 1: REAL
    """
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    cm_path = res_path / "confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (CIFAKE Test Split)")
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j,
                i,
                f"{val:,}",
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    return cm_path


def evaluate_model(
    model: tf.keras.Model,
    data_dir: str | Path,
    image_size: Tuple[int, int] = (160, 160),
    batch_size: int = 32,
    results_dir: str | Path = "results",
) -> Dict[str, Any]:
    """
    Runs full evaluation on the held-out test split.

    Args:
        model: Trained tf.keras.Model.
        data_dir: Path to dataset containing test/ folder.
        image_size: Input dimensions (H, W).
        batch_size: Batch size for testing.
        results_dir: Directory where metrics and figures will be saved.

    Returns:
        Dictionary of evaluated metrics.
    """
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON HELD-OUT TEST SPLIT")
    print("=" * 60)

    # Load test dataset strictly
    _, _, test_ds = load_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    # 1. Compute loss & built-in Keras metrics
    eval_results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    test_metrics = dict(zip(metric_names, [float(v) for v in eval_results]))

    # 2. Extract true labels and compute raw probability predictions
    y_true_list = []
    y_prob_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_prob_list.append(preds.flatten())
        y_true_list.append(labels.numpy().flatten())

    y_true = np.concatenate(y_true_list).astype(int)
    y_prob = np.concatenate(y_prob_list).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    # 3. Compute metrics via scikit-learn
    acc = float(accuracy_score(y_true, y_pred))
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 4. Format structured metrics
    results_payload = {
        "status": "COMPLETED",
        "dataset": "CIFAKE",
        "evaluation_split": "test",
        "total_test_samples": int(len(y_true)),
        "loss": float(test_metrics.get("loss", 0.0)),
        "accuracy": acc,
        "macro_avg": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1_score": float(f1_macro),
        },
        "per_class": {
            "FAKE (0)": {
                "precision": float(p_per_class[0]),
                "recall": float(r_per_class[0]),
                "f1_score": float(f1_per_class[0]),
                "support": int(support[0]),
            },
            "REAL (1)": {
                "precision": float(p_per_class[1]),
                "recall": float(r_per_class[1]),
                "f1_score": float(f1_per_class[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": {
            "order": ["FAKE (0)", "REAL (1)"],
            "matrix": cm.tolist(),
            "true_negatives_fake": int(cm[0, 0]),
            "false_positives_real": int(cm[0, 1]),
            "false_negatives_fake": int(cm[1, 0]),
            "true_positives_real": int(cm[1, 1]),
        },
    }

    # 5. Save metrics to JSON
    json_path = res_path / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved test metrics JSON to: {json_path.resolve()}")

    # 6. Save metrics to CSV
    csv_path = res_path / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric Category", "Class / Aggregate", "Precision", "Recall", "F1-Score", "Support"])
        writer.writerow(["Per-Class", "FAKE (0)", f"{p_per_class[0]:.4f}", f"{r_per_class[0]:.4f}", f"{f1_per_class[0]:.4f}", support[0]])
        writer.writerow(["Per-Class", "REAL (1)", f"{p_per_class[1]:.4f}", f"{r_per_class[1]:.4f}", f"{f1_per_class[1]:.4f}", support[1]])
        writer.writerow(["Aggregate", "Macro Average", f"{p_macro:.4f}", f"{r_macro:.4f}", f"{f1_macro:.4f}", len(y_true)])
        writer.writerow(["Overall", "Accuracy", f"{acc:.4f}", "-", "-", len(y_true)])
    print(f"Saved test metrics CSV to: {csv_path.resolve()}")

    # 7. Generate and save confusion matrix plot
    plot_path = plot_and_save_confusion_matrix(cm, results_dir=res_path)
    print(f"Saved confusion matrix plot to: {plot_path.resolve()}")

    # 8. Print textual summary
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")

    return results_payload
