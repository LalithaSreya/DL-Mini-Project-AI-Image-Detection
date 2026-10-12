# AI-Generated Image Detection Using Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Academic Institution**: Sreenidhi Institute of Science and Technology (Sreenidhi University)  
**Department**: Department of Computer Science and Engineering  
**Academic Year**: 2025–2026  
**GitHub Repository**: [LalithaSreya/DL-Mini-Project-AI-Image-Detection](https://github.com/LalithaSreya/DL-Mini-Project-AI-Image-Detection)  

---

## 1. Project Title
**AI-Generated Image Detection Using Deep Learning**

---

## 2. Project Overview
With the rapid advancements in deep generative modeling (e.g., Latent Diffusion Models, GANs), generating photorealistic synthetic imagery has become accessible to anyone. However, synthetic media also facilitates deceptive disinformation, malicious impersonation, and fraud.

This university mini-project builds, evaluates, and documents a robust binary image classification system using **Transfer Learning with MobileNetV2** on the standardized **CIFAKE** dataset to distinguish between authentic natural photographs and AI-generated synthetic images.

---

## 3. Problem Statement
Given an arbitrary input digital image $\mathbf{X}$, determine whether it is an authentic photographic capture (**REAL**, label `1`) or an AI-generated synthetic rendering (**FAKE**, label `0`), while maintaining computational efficiency, high classification accuracy, and academic reproducibility.

---

## 4. Project Objectives
1. **Automated Ingestion**: Validate, count, and stream images from the CIFAKE benchmark across training, validation, and held-out test splits.
2. **Transfer Learning**: Employ MobileNetV2 pretrained on ImageNet with a frozen backbone and customized classification head.
3. **Reproducible Pipeline**: Build modular scripts for training (`main.py` / `src/train.py`), evaluation (`src/evaluate.py`), and single-image prediction (`predict.py`).
4. **Verification**: Provide comprehensive unit tests (`tests/test_project.py`) and an interactive Google Colab notebook (`project.ipynb`).
5. **Academic Documentation**: Produce a complete university report draft in Markdown (`docs/REPORT_DRAFT.md`) and Microsoft Word (`docs/Project_Report.docx`).

---

## 5. Dataset Source
- **Dataset Name**: CIFAKE — Real and AI-Generated Synthetic Images
- **Official Kaggle Source**: [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Primary Publication**: Bird, J. J., & Lotfi, A. (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access, 12, 15600–15612.

---

## 6. Dataset Structure
The dataset contains a total of 120,000 images evenly balanced across two classes:
- **REAL (Label: 1)**: 60,000 authentic images from CIFAR-10 across 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **FAKE (Label: 0)**: 60,000 synthetic images generated via Stable Diffusion v1.4 conditioned on the same 10 category prompts.

### Expected Directory Layout
```text
data/CIFAKE/
├── train/
│   ├── FAKE/     (50,000 images)
│   └── REAL/     (50,000 images)
└── test/
    ├── FAKE/     (10,000 images)
    └── REAL/     (10,000 images)
```

---

## 7. Model Architecture
The detector is built upon **MobileNetV2** (Sandler et al., 2018):
- **Input Dimension**: $(160, 160, 3)$ raw pixel values in $[0, 255]$.
- **Integrated Preprocessing**: In-graph scaling to $[-1, 1]$ via `tf.keras.layers.Rescaling(scale=1/127.5, offset=-1.0)` matching MobileNetV2 preprocessing.
- **Pretrained Feature Extractor**: Base MobileNetV2 with ImageNet weights, frozen (`trainable = False`) to preserve low-level and mid-level feature representations.
- **Global Average Pooling 2D**: Collapses spatial dimensions $(5 \times 5 \times 1280)$ into a 1280-dimensional feature vector.
- **Dropout (0.2)**: Reduces co-adaptation and prevents overfitting.
- **Dense Classification Head**: Single output neuron with Sigmoid activation outputting probability $P(\text{REAL}) \in [0, 1]$.
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam ($\text{learning\_rate} = 10^{-4}$).

---

## 8. Transfer-Learning Explanation
Training deep neural networks from scratch on vision tasks requires massive datasets and substantial compute. Transfer learning addresses this by repurposing representations learned by a model trained on a large-scale visual benchmark (ImageNet: 1.4 million images across 1,000 classes).

By freezing the MobileNetV2 backbone, we preserve general feature filters (edges, textures, spatial gradients) and only train the newly added classification head. This strategy:
1. Prevents overfitting on smaller or specialized training distributions.
2. Drastically accelerates training convergence (typically requiring only 3–5 epochs).
3. Decreases computational overhead, enabling execution on consumer GPUs and Google Colab free tiers.

---

## 9. Technologies Used
- **Language**: Python 3.9+ (Tested on Python 3.14.2)
- **Deep Learning**: TensorFlow 2.x / Keras
- **Scientific Computing**: NumPy, SciPy
- **Data Visualization**: Matplotlib
- **Machine Learning Evaluation**: Scikit-Learn
- **Image Processing**: Pillow (PIL)
- **Testing**: Pytest
- **Reporting**: Python-Docx

---

## 10. Installation Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/LalithaSreya/DL-Mini-Project-AI-Image-Detection.git
cd DL-Mini-Project-AI-Image-Detection
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. Dataset Setup Instructions

### Option A: Manual Download (Recommended)
1. Download the zip archive from the [official Kaggle page](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
2. Extract the archive into `data/CIFAKE`:
   - Windows PowerShell:
     ```powershell
     Expand-Archive -Path "path\to\archive.zip" -DestinationPath "data\CIFAKE"
     ```
   - Linux / macOS / Colab:
     ```bash
     unzip archive.zip -d data/CIFAKE
     ```
3. Confirm that `data/CIFAKE/train/` and `data/CIFAKE/test/` contain `FAKE/` and `REAL/` folders.

### Option B: Kaggle API CLI
```bash
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/
unzip data/cifake-real-and-ai-generated-synthetic-images.zip -d data/CIFAKE
```

---

## 12. Training Commands
To start model training with default hyperparameters:
```bash
python main.py --data_dir data/CIFAKE --epochs 5 --batch_size 32
```

### Configurable Hyperparameter Arguments:
| Argument | Default | Description |
| :--- | :---: | :--- |
| `--data_dir` | `data/CIFAKE` | Path to extracted CIFAKE directory |
| `--epochs` | `5` | Maximum training epochs |
| `--batch_size` | `32` | Training mini-batch size |
| `--image_size` | `160` | Input dimension (height and width) |
| `--learning_rate` | `0.0001` | Adam optimizer learning rate |
| `--models_dir` | `models` | Directory to save checkpoint weights |
| `--results_dir` | `results` | Directory to save metric tables and plots |
| `--validation_split` | `0.2` | Fraction of train data reserved for validation |
| `--seed` | `42` | Random seed for reproducible splitting |

---

## 13. Evaluation Commands
To evaluate an existing trained model checkpoint without retraining:
```bash
python main.py --data_dir data/CIFAKE --eval_only --model_path models/mobilenetv2_cifake_best.keras
```

---

## 14. Single-Image Prediction Commands
To test a single image file (`.jpg`, `.jpeg`, `.png`):
```bash
python predict.py --image path/to/sample.jpg
```

Or specify a custom model checkpoint:
```bash
python predict.py --image path/to/sample.jpg --model_path models/mobilenetv2_cifake_best.keras
```

### Example Output:
```text
==================================================
      IMAGE DETECTION INFERENCE RESULT        
==================================================
Input Image     : sample.jpg
Predicted Class : FAKE
Confidence      : 96.42%
Raw Model Score : 0.0358 (Sigmoid P(REAL))
--------------------------------------------------
Class Interpretation:
  - Score >= 0.50 -> REAL (Natural photograph)
  - Score <  0.50 -> FAKE (AI-generated synthetic image)
==================================================
```

---

## 15. Output Files and Generated Artifacts
When the training and evaluation pipelines run, the following files are produced:

```text
models/
├── mobilenetv2_cifake_best.keras    # Optimal model weights (min val_loss)
├── mobilenetv2_cifake_final.keras   # Final epoch weights
└── training_history.json            # Epoch-by-epoch loss and accuracy metrics

results/
├── metrics.json                     # Structured test evaluation results
├── metrics.csv                      # Tabular per-class & macro metrics
├── training_loss_curve.png          # Train vs. Validation Loss plot
├── training_accuracy_curve.png      # Train vs. Validation Accuracy plot
└── confusion_matrix.png             # Annotated test confusion matrix
```

---

## 16. Google Colab Instructions
A complete, sequential Google Colab workflow is provided in [`project.ipynb`](project.ipynb):
1. Open [Google Colab](https://colab.research.google.com/) and upload `project.ipynb`.
2. Navigate to **Runtime** > **Change runtime type**, select **T4 GPU**, and click Save.
3. Follow the notebook cells sequentially:
   - Cell 1–2: Environment setup and library installation.
   - Cell 3–4: Dataset path configuration (edit `DATA_DIR` to your path).
   - Cell 5–7: Dataset loading and model construction.
   - Cell 8–10: Training, metric visualization, and test evaluation.
   - Cell 11: Single-image prediction with sample visual demonstration.

---

## 17. Results Section & Academic Honesty Disclosure
> **STATUS: EXPERIMENTAL RUN PENDING REAL EXECUTION**
>
> In accordance with academic honesty principles, this repository does **NOT** present fabricated, pre-computed, or plagiarized metrics. The codebase, unit tests (12 passing tests), preprocessing pipelines, model definition, and inference interfaces are fully implemented and verified.
>
> To generate genuine experimental results, run the pipeline on an extracted CIFAKE dataset:
> ```bash
> python main.py --data_dir data/CIFAKE --epochs 5 --batch_size 32
> ```
> Based on published findings by Bird & Lotfi (2023), baseline CNN architectures on CIFAKE achieve ~92.9% accuracy. The MobileNetV2 transfer-learning model in this repository is expected to achieve comparable or superior performance (~92%–94%) when trained on the full 100,000 training images.

---

## 18. Limitations
1. **Generator Specificity**: CIFAKE's synthetic class was generated exclusively with **Stable Diffusion v1.4**. While the model learns features representative of diffusion synthesis, it may exhibit reduced sensitivity against other generators (e.g., Midjourney v6, DALL-E 3, Flux) or GAN architectures.
2. **Resolution Downsampling**: CIFAR-10 images are natively $32 \times 32$ pixels. Resizing to $160 \times 160$ allows MobileNetV2 feature extraction but does not incorporate high-resolution $1024 \times 1024$ native artifacts.
3. **Compression & Social Media Re-encoding**: Severe JPEG compression or platform filtering can obscure fine-grained generative artifacts, altering model confidence.

---

## 19. References
1. **Bird, J. J., & Lotfi, A.** (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access, 12, 15600–15612.
2. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C.** (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of CVPR, 4510–4520.
3. **Krizhevsky, A.** (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto.
4. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. Proceedings of CVPR, 10684–10695.
5. **Wang, S. Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A.** (2020). *CNN-Generated Images Are Surprisingly Easy to Spot... for Now*. Proceedings of CVPR, 8695–8704.

---

## 20. Team Members & Project Contributors
- **Student 1**: [Student Name Placeholder 1] (Roll No: `[21031AXXXX]`)
- **Student 2**: [Student Name Placeholder 2] (Roll No: `[21031AXXXX]`)
- **Student 3**: [Student Name Placeholder 3] (Roll No: `[21031AXXXX]`)

**Project Guide**: [Faculty Guide Name Placeholder], Assistant Professor / Associate Professor, Department of Computer Science and Engineering, Sreenidhi Institute of Science and Technology.
