 # AI-Generated Image Detection Using Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Academic Institution**: SREENIDHI UNIVERSITY  
**Department**: Department of Computer Science and Engineering  
**Academic Year**: 2026–2027  
**Course Coordinator**: Dr. Saritha Bantu, Professor, Department of CSE, Sreenidhi University  
**Team Members**:
- **Bojja Sahasra** (`2401100CSE0243`)
- **Mudigonda Lalitha Sreya** (`2401100CSE0264`)  
- **Rohan Joshua Francis** (`2401100CSE0275`)
**GitHub Repository**: [LalithaSreya/DL-Mini-Project-AI-Image-Detection](https://github.com/LalithaSreya/DL-Mini-Project-AI-Image-Detection)  

---

## 1. Project Title
**AI-Generated Image Detection Using Deep Learning**

---

## 2. Project Overview
With the rapid advancements in deep generative modeling (e.g., Latent Diffusion Models, GANs), generating photorealistic synthetic imagery has become accessible to anyone. However, synthetic media also facilitates deceptive disinformation, malicious impersonation, and fraud.

This university mini-project builds, evaluates, and documents a robust binary image classification system using **Transfer Learning with MobileNetV2** on the standardized **CIFAKE** dataset to distinguish between authentic natural photographs and AI-generated synthetic images.

### Directory Organization: Submission vs. Preparation
To maintain clean separation between university deliverables and student study materials, the project is structured as follows:
- 📁 **[`SUBMISSION/`](SUBMISSION/) (Official University Package)**: Contains the clean, self-contained academic deliverables intended for faculty evaluation, including the formal report (Word & PDF), verified Colab notebook, web demo app, requirements, source modules, and demonstration guide.
- 📁 **[`PREPARATION/`](PREPARATION/) (Personal Student Materials)**: Contains private preparation notes, technical viva Q&A, report review checklist, model verification audit, and internal diagnostic logs for personal study and oral exam defense.
- 📁 **Project Root (Development Environment)**: Contains the full working codebase, test suites, original and repaired notebooks, and local model checkpoints.

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

## 15. Interactive Web Application (Streamlit GUI)
For live demonstrations, presentations, and faculty reviews, launch the interactive graphical web interface:

```bash
streamlit run app.py
```

### Features of the Web Interface:
1. **Drag-and-Drop Image Testing**: Upload any custom `.jpg`, `.png`, or `.jpeg` file for instant inference.
2. **Benchmark Test Samples**: Select from built-in authentic and synthetic CIFAKE test images with a single click.
3. **Visual Confidence Gauges**: Real-time progress bars showing $P(\text{REAL})$ vs. $P(\text{FAKE})$ probabilities.
4. **Interactive Metrics Dashboard**: View test accuracy, precision, recall, F1 tables, confusion matrix heatmap, and training curves directly in the browser.
5. **Adjustable Decision Threshold**: Real-time slider to test sensitivity variations.

---

## 16. Output Files and Generated Artifacts
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

## 17. Google Colab Instructions
A complete, sequential, and fully verified Google Colab workflow is provided in [`project_colab_fixed.ipynb`](project_colab_fixed.ipynb) (with the original notebook preserved at [`project.ipynb`](project.ipynb)):
1. Open [Google Colab](https://colab.research.google.com/) and upload [`project_colab_fixed.ipynb`](project_colab_fixed.ipynb).
2. Navigate to **Runtime** > **Change runtime type**, select **T4 GPU**, and click Save.
3. Follow the notebook cells sequentially:
   - Cell 1–2: Environment setup, GPU verification, and library installation (`kagglehub`, `tensorflow`, `matplotlib`, `scikit-learn`).
   - Cell 3–4: Automated dataset download via `kagglehub` and directory validation.
   - Cell 5–7: Dataset pipeline construction with tf.data streaming and MobileNetV2 model construction.
   - Cell 8–10: GPU training (5 epochs), training history visualization, and test evaluation.
   - Cell 11: Prediction demonstration on unseen test samples.
4. Download the trained artifacts (`mobilenetv2_cifake_best.keras`, `training_history.json`, figures, and metrics) from Google Colab into your local project repository under `models/` and `results/`.

---

## 18. Academic Deliverables and Submission Documentation
All academic documentation conforming strictly to Sreenidhi University B.Tech guidelines has been organized into dedicated directories:

### Official Submission Package (`SUBMISSION/`)
- **Official Project Report (Word DOCX)**: [`SUBMISSION/report/AI_Generated_Image_Detection_Project_Report.docx`](SUBMISSION/report/AI_Generated_Image_Detection_Project_Report.docx)
- **Official Project Report (PDF)**: [`SUBMISSION/report/AI_Generated_Image_Detection_Project_Report.pdf`](SUBMISSION/report/AI_Generated_Image_Detection_Project_Report.pdf)
- **Interactive Demonstration Guide**: [`SUBMISSION/docs/demonstration_guide.md`](SUBMISSION/docs/demonstration_guide.md) — 5–8 minute viva presentation plan and web app demonstration.
- **Verified Colab Training Notebook**: [`SUBMISSION/project_colab_fixed.ipynb`](SUBMISSION/project_colab_fixed.ipynb) — Executable Google Colab GPU workflow.
- **Interactive Web App**: [`SUBMISSION/app.py`](SUBMISSION/app.py) — Live Streamlit GUI.
- **Experimental Artifacts**: [`SUBMISSION/results/`](SUBMISSION/results/) — Figures and metrics.

### Personal Student Preparation Package (`PREPARATION/`)
- **Viva Voce Technical Q&A**: [`PREPARATION/viva_questions.md`](PREPARATION/viva_questions.md) — 14 in-depth questions and answers covering architectures, transfer learning, metrics, and failure modes.
- **Report Review Checklist**: [`PREPARATION/report_review_checklist.md`](PREPARATION/report_review_checklist.md) — Itemized audit checklist with verification status.
- **Final Submission Verification Checklist**: [`PREPARATION/final_submission_checklist.md`](PREPARATION/final_submission_checklist.md) — 20-item comprehensive compliance matrix.
- **Model Verification Audit**: [`PREPARATION/model_verification.md`](PREPARATION/model_verification.md) — Detailed technical audit of checkpoints, mappings, and evaluation metrics.
- **Internal Engineering Notes**: [`PREPARATION/internal_notes.md`](PREPARATION/internal_notes.md) — Confidential notes on accuracy bugs, hardware constraints, and viva defense talking points.

---

## 19. Results Section & Academic Honesty Disclosure
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

## 20. Limitations
1. **Generator Specificity**: CIFAKE's synthetic class was generated exclusively with **Stable Diffusion v1.4**. While the model learns features representative of diffusion synthesis, it may exhibit reduced sensitivity against other generators (e.g., Midjourney v6, DALL-E 3, Flux) or GAN architectures.
2. **Resolution Downsampling**: CIFAR-10 images are natively $32 \times 32$ pixels. Resizing to $160 \times 160$ allows MobileNetV2 feature extraction but does not incorporate high-resolution $1024 \times 1024$ native artifacts.
3. **Compression & Social Media Re-encoding**: Severe JPEG compression or platform filtering can obscure fine-grained generative artifacts, altering model confidence.

---

## 21. Future Scope
1. **Multi-Generator Generalization**: Extend evaluation and fine-tuning across diverse state-of-the-art diffusion and autoregressive generators (e.g., Midjourney v6, DALL-E 3, Adobe Firefly, Flux.1, and Sora) to eliminate generator-specific overfitting.
2. **Explainable AI (XAI)**: Integrate Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps into the Streamlit interface to visualize precisely which spatial regions (e.g. textural inconsistencies, warped boundaries, unnatural reflections) triggered the synthetic classification.
3. **Frequency-Domain & Multi-Modal Analysis**: Incorporate 2D Discrete Cosine Transform (DCT) and Fast Fourier Transform (FFT) spectral analysis to detect high-frequency artifacts left by upsampling and deconvolution operations.
4. **Production API & Browser Extension**: Package the lightweight MobileNetV2 model into a low-latency FastAPI microservice and client-side browser extension for real-time verification of visual media across web platforms.

---

## 22. References
1. **Bird, J. J., & Lotfi, A.** (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access, 12, 15600–15612.
2. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C.** (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of CVPR, 4510–4520.
3. **Krizhevsky, A.** (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto.
4. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. Proceedings of CVPR, 10684–10695.
5. **Wang, S. Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A.** (2020). *CNN-Generated Images Are Surprisingly Easy to Spot... for Now*. Proceedings of CVPR, 8695–8704.

---

## 23. Team Members & Course Coordinator
- **Bojja Sahasra** (Roll No: `2401100CSE0243`)
- **Mudigonda Lalitha Sreya** (Roll No: `2401100CSE0264`)
- **Rohan Joshua Francis** (Roll No: `2401100CSE0275`)

**Course Coordinator / Project Guide**: **Dr. Saritha Bantu**, Professor, Department of Computer Science and Engineering, Sreenidhi University.
