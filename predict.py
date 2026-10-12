"""
Single-Image Inference for AI-Generated Image Detection.
Evaluates an individual image file and predicts whether it is REAL or FAKE.

Class Mapping:
    0: FAKE (AI-generated synthetic image)
    1: REAL (Authentic natural image)

Example Usage:
    python predict.py --image path/to/sample.jpg
    python predict.py --image path/to/sample.jpg --model_path models/mobilenetv2_cifake_best.keras
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from src.dataset import CLASS_NAMES, LABEL_TO_CLASS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-image prediction using the trained MobileNetV2 CIFAKE detector."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file (.jpg, .jpeg, .png, etc.).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/mobilenetv2_cifake_best.keras",
        help="Path to the trained Keras model file (default: 'models/mobilenetv2_cifake_best.keras').",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=160,
        help="Input image dimension expected by the model (default: 160).",
    )
    return parser.parse_args()


def load_and_preprocess_image(image_path: str | Path, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Validates, loads, and formats an individual image for model inference.

    Args:
        image_path: Path to the image file.
        target_size: (Height, Width) dimensions.

    Returns:
        np.ndarray: Batch tensor of shape (1, target_size[0], target_size[1], 3) with values in [0, 255].

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file is corrupted or not a valid image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: '{path.resolve()}'.")

    try:
        with Image.open(path) as img:
            # Convert to standard RGB (handles grayscale, RGBA, palette modes safely)
            img = img.convert("RGB")
            # Resize using bilinear resampling matching tf.keras defaults
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            img_array = np.array(img, dtype=np.float32)
            # Expand to batch dimension: (1, H, W, 3)
            batch_tensor = np.expand_dims(img_array, axis=0)
            return batch_tensor
    except UnidentifiedImageError:
        raise ValueError(f"File is not a supported or decodable image: '{path.resolve()}'.")
    except Exception as e:
        raise ValueError(f"Failed to read image at '{path.resolve()}': {e}")


def predict_image(
    image_path: str | Path,
    model: tf.keras.Model,
    target_size: Tuple[int, int] = (160, 160),
) -> Tuple[str, float, float]:
    """
    Runs model inference on an input image.

    Returns:
        (predicted_class, real_probability, confidence_score)
        - predicted_class: 'REAL' or 'FAKE'
        - real_probability: Raw sigmoid output P(REAL) in [0.0, 1.0]
        - confidence_score: Confidence percentage for the predicted label in [50.0, 100.0]
    """
    tensor = load_and_preprocess_image(image_path, target_size=target_size)
    pred_prob = float(model.predict(tensor, verbose=0)[0][0])

    if pred_prob >= 0.5:
        predicted_class = "REAL"
        confidence = pred_prob * 100.0
    else:
        predicted_class = "FAKE"
        confidence = (1.0 - pred_prob) * 100.0

    return predicted_class, pred_prob, confidence


def main():
    args = parse_args()

    # 1. Locate trained model
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Check fallback to final model if best doesn't exist
        fallback_path = Path("models/mobilenetv2_cifake_final.keras")
        if fallback_path.exists():
            model_path = fallback_path
        else:
            print("\n" + "!" * 70)
            print("[ERROR] Trained model file not found!")
            print(f"Looked for: '{Path(args.model_path).resolve()}'")
            print("\nPlease train the model first before performing predictions:")
            print("  python main.py --data_dir data/CIFAKE --epochs 5")
            print("!" * 70)
            sys.exit(1)

    # 2. Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n[ERROR] Image file does not exist: '{image_path.resolve()}'")
        sys.exit(1)

    # 3. Load model
    print(f"Loading detector model from: {model_path.resolve()}...")
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 4. Run prediction
    try:
        predicted_class, raw_score, confidence = predict_image(
            image_path=image_path,
            model=model,
            target_size=(args.image_size, args.image_size),
        )
    except Exception as err:
        print(f"[ERROR] Inference failed: {err}")
        sys.exit(1)

    # 5. Format output
    print("\n" + "=" * 50)
    print("      IMAGE DETECTION INFERENCE RESULT        ")
    print("=" * 50)
    print(f"Input Image     : {image_path.name}")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence:.2f}%")
    print(f"Raw Model Score : {raw_score:.4f} (Sigmoid P(REAL))")
    print("-" * 50)
    print("Class Interpretation:")
    print("  - Score >= 0.50 -> REAL (Natural photograph)")
    print("  - Score <  0.50 -> FAKE (AI-generated synthetic image)")
    print("=" * 50)
    print(
        "\nNote: Deep learning confidence reflects model similarity to the CIFAKE training"
        "\ndistribution (Stable Diffusion v1.4 vs CIFAR-10) and does not constitute absolute"
        "\nforensic proof across unseen generative models."
    )


if __name__ == "__main__":
    main()
