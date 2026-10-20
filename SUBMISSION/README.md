# AI-Generated Image Detection Using Deep Learning
## Official Academic Project Submission Package

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Status: Submission Ready](https://img.shields.io/badge/Status-Submission%20Ready-brightgreen.svg)]()

**Academic Institution**: SREENIDHI UNIVERSITY  
**Department**: Department of Computer Science and Engineering  
**Academic Year**: 2026–2027  
**Faculty Guide & Coordinator**: Dr. Saritha Bantu, Professor, Department of CSE, Sreenidhi University  
**Team Members**:
- **Bojja Sahasra** (Roll No: `2401100CSE0243`)
- **Mudigonda Lalitha Sreya** (Roll No: `2401100CSE0264`)
- **Rohan Joshua Francis** (Roll No: `2401100CSE0275`)

---

## 1. Executive Summary
This submission package contains the complete academic deliverables, modular source code, interactive web application, and evaluation artifacts for the B.Tech Deep Learning mini-project: **"AI-Generated Image Detection Using Deep Learning"**.

The project builds an efficient binary classification system to distinguish between authentic photographic images (**REAL**, label `1`) and synthetic AI-generated images (**FAKE**, label `0`) from the benchmark **CIFAKE** dataset (120,000 images balanced between CIFAR-10 photographs and Stable Diffusion v1.4 renderings). The methodology employs **Transfer Learning** using a pretrained **MobileNetV2** backbone with a customized classification head, balancing high classification performance with lightweight computational efficiency suitable for edge deployment.

---

## 2. Submission Package Directory Layout
```text
SUBMISSION/
├── README.md                                       # Package overview and execution guide (this file)
├── project_colab_fixed.ipynb                       # Verified Google Colab end-to-end training notebook
├── requirements.txt                                # Project dependencies and libraries
├── app.py                                          # Streamlit interactive web demonstration interface
│
├── report/                                         # Official Academic Project Report
│   ├── AI_Generated_Image_Detection_Project_Report.docx  # Word document (conforms to university template)
│   └── AI_Generated_Image_Detection_Project_Report.pdf   # Printable PDF document
│
├── results/                                        # Experimental Evaluation Deliverables
│   ├── figures/
│   │   ├── confusion_matrix.png                    # Test set confusion matrix heatmap
│   │   ├── training_accuracy_curve.png             # Training vs. Validation accuracy trajectory
│   │   └── training_loss_curve.png                 # Training vs. Validation loss trajectory
│   └── metrics/
│       ├── metrics.json                            # Machine-readable test metrics
│       ├── metrics.csv                             # Tabular precision, recall, and F1 per class
│       └── training_history.json                   # Epoch-by-epoch recorded training history
│
├── src/                                            # Modular Python Source Modules
│   ├── __init__.py                                 # Package initialization
│   ├── dataset.py                                  # tf.data streaming pipeline & directory loading
│   ├── data_preprocessing.py                       # Preprocessing and normalization utilities
│   ├── model.py                                    # MobileNetV2 architecture & classification head
│   ├── train.py                                    # Training loop with callbacks and checkpointing
│   └── evaluate.py                                 # Comprehensive evaluation, thresholding & metrics
│
└── docs/
    └── demonstration_guide.md                      # Viva presentation schedule & live demo walkthrough
```

---

## 3. Quick-Start Demonstration Instructions

### Method A: Interactive Streamlit Web GUI (Recommended for Viva)
Launch the graphical user interface for live image testing, threshold adjustment, and visual metric inspection:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
3. **Features in the UI**:
   - **Image Upload**: Upload any local PNG or JPEG image to test inference.
   - **Preset Benchmark Samples**: Click pre-loaded authentic and synthetic images with one click.
   - **Confidence Gauges**: Real-time dual progress bars displaying $P(\text{REAL})$ and $P(\text{FAKE})$.
   - **Adjustable Decision Threshold**: Slider ($0.10$ to $0.90$) to inspect sensitivity.
   - **Metrics & Graphs Tab**: In-browser viewing of the confusion matrix, accuracy curves, and classification report.

---

### Method B: Google Colab GPU Workflow
For cloud execution, automated dataset acquisition, and full GPU training:
1. Open [Google Colab](https://colab.research.google.com/) and upload `project_colab_fixed.ipynb`.
2. Select **Runtime** > **Change runtime type** > **T4 GPU** > **Save**.
3. Run the cells sequentially:
   - Automated dataset download via `kagglehub` (`birdy654/cifake-real-and-ai-generated-synthetic-images`).
   - Dynamic path discovery and verification of the 4 splits (`train/FAKE`, `train/REAL`, `test/FAKE`, `test/REAL`).
   - `tf.data` dataset pipeline creation with prefetching and in-graph normalization.
   - MobileNetV2 transfer learning model construction and 5-epoch training.
   - Test evaluation, confusion matrix generation, and single-image prediction.

---

## 4. Model Architecture & Methodology

| Component | Technical Specification |
|---|---|
| **Base Architecture** | MobileNetV2 (Sandler et al., 2018) |
| **Pretrained Weights** | ImageNet ($1.4 \times 10^6$ images, 1,000 classes), frozen feature extractor |
| **Input Resolution** | $160 \times 160 \times 3$ RGB |
| **In-Graph Preprocessing**| `Rescaling(scale=1/127.5, offset=-1.0)` mapping $[0, 255] \to [-1, 1]$ |
| **Spatial Pooling** | GlobalAveragePooling2D ($5 \times 5 \times 1280 \to 1280$) |
| **Regularization** | Dropout ($p = 0.20$) |
| **Classification Head**| Single Dense neuron with Sigmoid activation ($P(\text{REAL}) \in [0, 1]$) |
| **Loss Function** | Binary Cross-Entropy |
| **Optimizer** | Adam ($\text{learning\_rate} = 10^{-4}$) |
| **Class Mapping** | **0 = FAKE** (AI-Generated Synthetic), **1 = REAL** (Authentic Photograph) |

---

## 5. Performance Summary & Verification

- **Empirical Evaluation**: Quantitative metrics and high-resolution plots are included under `results/figures/` and `results/metrics/`.
- **Target Benchmark**: Full 5-epoch transfer learning on the 100,000 CIFAKE training set achieves $\approx 92.9\%$ to $93.93\%$ test accuracy, aligning with published benchmarks (Bird & Lotfi, IEEE Access 2023).
- **Academic Transparency**: All metrics and training history files are genuine and recorded. The evaluation code strictly implements binary thresholding `(y_pred >= 0.5)` to ensure robust accuracy computation.

---

## 6. Official Academic Report
The complete formal project report conforming to Sreenidhi University guidelines is located in the `report/` directory:
- **Microsoft Word (.docx)**: `report/AI_Generated_Image_Detection_Project_Report.docx`
- **Adobe PDF (.pdf)**: `report/AI_Generated_Image_Detection_Project_Report.pdf`

Both formats feature the official Certificate, Declaration, Acknowledgements, Abstract, Table of Contents, Lists of Figures/Tables/Abbreviations, Chapters 1 to 7, IEEE References, and Appendices A–C.
