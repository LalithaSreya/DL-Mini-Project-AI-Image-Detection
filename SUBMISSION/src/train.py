"""
Training pipeline for MobileNetV2 on the CIFAKE dataset.
Ensures:
- Training split is partitioned into train and validation sets.
- Test split is kept separate and held out.
- Checkpoints and training histories are saved reliably.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import tensorflow as tf

from src.dataset import load_datasets, count_dataset_images
from src.model import build_mobilenetv2_classifier


def run_training(
    data_dir: str | Path,
    output_dir: str | Path = "models",
    image_size: Tuple[int, int] = (160, 160),
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 5,
    validation_split: float = 0.2,
    seed: int = 42,
    patience: int = 3,
    weights: str = "imagenet",
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Executes the training workflow.

    Args:
        data_dir: Path to dataset folder containing train/ and test/.
        output_dir: Destination folder for model checkpoints.
        image_size: (Height, Width) input dimensions.
        batch_size: Number of images per training batch.
        learning_rate: Adam optimizer learning rate.
        epochs: Maximum number of epochs to train.
        validation_split: Fraction of train data reserved for validation.
        seed: Random seed for reproducibility.
        patience: Early stopping patience epochs.
        weights: Base pretrained weights ('imagenet' or None).
        steps_per_epoch: Number of steps (batches) per epoch (None for full split).
        validation_steps: Number of validation steps (None for full split).

    Returns:
        (trained_model, history_dict)
    """
    # Set random seeds
    tf.keras.utils.set_random_seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AI-GENERATED IMAGE DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset root: {Path(data_dir).resolve()}")
    print(f"Image dimensions: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Validation split: {validation_split * 100:.0f}% of train folder")
    print(f"Output directory: {out_path.resolve()}")

    # 1. Count images and display distribution
    counts = count_dataset_images(data_dir)
    print("\nDataset Summary:")
    print(f"  Train: FAKE={counts['train']['FAKE']}, REAL={counts['train']['REAL']} (Total: {counts['train']['total']})")
    print(f"  Test : FAKE={counts['test']['FAKE']}, REAL={counts['test']['REAL']} (Total: {counts['test']['total']})")

    # 2. Load data splits
    print("\nLoading dataset pipelines...")
    train_ds, val_ds, _ = load_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    # 3. Build model
    print("\nBuilding MobileNetV2 transfer-learning model...")
    model = build_mobilenetv2_classifier(
        input_shape=(image_size[0], image_size[1], 3),
        learning_rate=learning_rate,
        weights=weights,
        freeze_backbone=True,
    )
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # 4. Configure callbacks
    best_checkpoint_path = out_path / "mobilenetv2_cifake_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 5. Execute training
    print("\nStarting model training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,
        )
    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        raise

    # 6. Save final model and history log
    final_model_path = out_path / "mobilenetv2_cifake_final.keras"
    model.save(str(final_model_path))
    print(f"\nSaved final model checkpoint to: {final_model_path.resolve()}")
    if best_checkpoint_path.exists():
        print(f"Best model checkpoint saved to: {best_checkpoint_path.resolve()}")

    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_path = out_path / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path.resolve()}")

    return model, history_dict
