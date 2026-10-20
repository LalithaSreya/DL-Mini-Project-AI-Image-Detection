"""
Model architecture definition for AI-Generated Image Detection.
Implements transfer learning using MobileNetV2 with ImageNet pretrained weights.
Class mapping:
    0: FAKE (AI-generated synthetic image)
    1: REAL (Authentic natural image)
Output:
    Single sigmoid neuron predicting probability that input image is REAL.
    P(REAL) >= 0.5 -> REAL (1)
    P(REAL) < 0.5  -> FAKE (0)
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (160, 160, 3),
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.2,
    weights: Optional[str] = "imagenet",
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """
    Constructs and compiles a transfer-learning binary classification model based on MobileNetV2.

    Architecture:
        1. Input Layer: shape (160, 160, 3) accepting raw pixel values in [0, 255].
        2. Preprocessing Layer: mobilenet_v2.preprocess_input (scales to [-1, 1]).
        3. MobileNetV2 Base: Frozen pretrained feature extractor (include_top=False).
        4. GlobalAveragePooling2D: Flattens spatial feature maps to a 1D vector.
        5. Dropout: Regularization to combat overfitting.
        6. Dense(1, activation='sigmoid'): Outputs probability P(REAL).

    Args:
        input_shape: Image dimensions (Height, Width, Channels). Default (160, 160, 3).
        learning_rate: Adam optimizer learning rate. Default 1e-4.
        dropout_rate: Dropout fraction before the classification head. Default 0.2.
        weights: Pretrained weights to load, e.g. 'imagenet' or None for random initialization.
        freeze_backbone: If True, freezes all weights in the MobileNetV2 base extractor.

    Returns:
        Compiled tf.keras.Model ready for training and evaluation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Built-in MobileNetV2 preprocessing maps [0, 255] float/int to [-1, 1] via (x / 127.5) - 1.0.
    # Using layers.Rescaling ensures exact consistency in training, evaluation,
    # and single-image inference while supporting seamless model saving and loading across Keras versions.
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_preprocessing",
    )(inputs)

    # Base feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    if freeze_backbone:
        base_model.trainable = False
    else:
        base_model.trainable = True

    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_cifake_detector")

    # Compile with Adam and BinaryCrossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
