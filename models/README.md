# Models Directory

This directory stores trained deep learning model weights and checkpoints generated during training runs.

## Expected Contents

When the training pipeline (`main.py` or `src/train.py`) executes successfully, model checkpoints are saved here:

- `mobilenetv2_cifake_best.keras`: Best model weights saved via `ModelCheckpoint` based on validation loss.
- `mobilenetv2_cifake_final.keras`: Model saved at the end of all training epochs.

## Model Details

- **Architecture**: MobileNetV2 Transfer Learning
- **Pretrained Weights**: ImageNet
- **Input Dimensions**: `(160, 160, 3)`
- **Output**: Single sigmoid unit representing binary probability
- **Class Mapping**:
  - `0`: **FAKE** (AI-Generated Synthetic Image)
  - `1`: **REAL** (Authentic Natural Image)

## Note on Git Tracking

Large binary model files (`*.keras`, `*.h5`) are excluded from Git commits via `.gitignore` to prevent exceeding GitHub file size limits and repository bloat. To generate the model locally, follow the training instructions in the root [README.md](../README.md).
