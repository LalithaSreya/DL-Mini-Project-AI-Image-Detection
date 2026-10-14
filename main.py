"""
AI-Generated Image Detection Using Deep Learning
Main entry point for training and evaluation.

Example usage:
    # Full training and evaluation:
    python main.py --data_dir data/CIFAKE --epochs 5 --batch_size 32

    # Evaluate only on an existing model:
    python main.py --data_dir data/CIFAKE --eval_only --model_path models/mobilenetv2_cifake_best.keras
"""

import argparse
import sys
from pathlib import Path
import tensorflow as tf

from src.dataset import validate_dataset_dir, DatasetNotFoundError, InvalidDatasetStructureError
from src.train import run_training
from src.evaluate import evaluate_model, plot_and_save_history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate MobileNetV2 for AI-Generated Image Detection on the CIFAKE dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/CIFAKE",
        help="Path to the extracted CIFAKE dataset containing train/ and test/ folders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for training and testing (default: 32).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=160,
        help="Input image height and width in pixels (default: 160).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Adam optimizer learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints (default: 'models').",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation metrics and plots (default: 'results').",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Fraction of training images reserved for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Number of batch steps per training epoch (default: None for full dataset).",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Number of validation batch steps (default: None for full dataset).",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=None,
        help="Number of test batch steps for evaluation (default: None for full dataset).",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only evaluate an existing saved model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model file when running in --eval_only mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("   AI-GENERATED IMAGE DETECTION USING DEEP LEARNING (CIFAKE)   ")
    print("   SREENIDHI UNIVERSITY                                        ")
    print("   Department of Computer Science and Engineering              ")
    print("=" * 70)

    # 1. Validate dataset existence before proceeding
    data_path = Path(args.data_dir)
    try:
        validate_dataset_dir(data_path)
        print(f"[OK] Verified dataset directory structure at: {data_path.resolve()}\n")
    except (DatasetNotFoundError, InvalidDatasetStructureError) as err:
        print("\n" + "!" * 70)
        print(f"[DATASET ERROR] {err}")
        print("!" * 70)
        print("\nPlease follow the instructions in 'data/README.md' to download and extract CIFAKE:")
        print("  1. Download CIFAKE from: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("  2. Extract the archive into your data folder so the structure is:")
        print("     data/CIFAKE/train/FAKE, data/CIFAKE/train/REAL")
        print("     data/CIFAKE/test/FAKE,  data/CIFAKE/test/REAL")
        print("  3. Re-run: python main.py --data_dir data/CIFAKE")
        print("!" * 70)
        sys.exit(1)

    img_size = (args.image_size, args.image_size)

    # 2. Execution Mode: Evaluation Only
    if args.eval_only:
        target_model_path = args.model_path or str(Path(args.models_dir) / "mobilenetv2_cifake_best.keras")
        model_file = Path(target_model_path)
        if not model_file.exists():
            print(f"\n[ERROR] Model checkpoint not found at: '{model_file.resolve()}'.")
            print("Please train a model first using: python main.py --data_dir data/CIFAKE")
            sys.exit(1)

        print(f"Loading trained model from: {model_file.resolve()}...")
        model = tf.keras.models.load_model(str(model_file))
        evaluate_model(
            model=model,
            data_dir=data_path,
            image_size=img_size,
            batch_size=args.batch_size,
            results_dir=args.results_dir,
            test_steps=args.test_steps,
        )
        print("\nEvaluation completed successfully.")
        return

    # 3. Execution Mode: Train + Evaluate
    model, history = run_training(
        data_dir=data_path,
        output_dir=args.models_dir,
        image_size=img_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        validation_split=args.validation_split,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )

    # Generate and save loss/accuracy curves
    print("\nGenerating training & validation history curves...")
    loss_fig, acc_fig = plot_and_save_history(history, results_dir=args.results_dir)
    print(f"  Loss curve saved: {loss_fig.resolve()}")
    print(f"  Accuracy curve saved: {acc_fig.resolve()}")

    # Evaluate on held-out test split
    best_model_path = Path(args.models_dir) / "mobilenetv2_cifake_best.keras"
    if best_model_path.exists():
        print(f"\nLoading best checkpoint for test split evaluation: {best_model_path.resolve()}")
        eval_model = tf.keras.models.load_model(str(best_model_path))
    else:
        eval_model = model

    evaluate_model(
        model=eval_model,
        data_dir=data_path,
        image_size=img_size,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        test_steps=args.test_steps,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Model saved to:   {Path(args.models_dir).resolve()}")
    print(f"Results saved to: {Path(args.results_dir).resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
