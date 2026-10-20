# Project Demonstration Guide (5–8 Minute Presentation)

**Project Title**: AI-Generated Image Detection Using Deep Learning  
**Department**: Computer Science and Engineering | Sreenidhi University  
**Academic Year**: 2026–2027  
**Team Members**:
- Bojja Sahasra (`2401100CSE0243`)
- Mudigonda Lalitha Sreya (`2401100CSE0264`)
- Rohan Joshua Francis (`2401100CSE0275`)  
**Faculty Guide**: Dr. Saritha Bantu, Professor, Department of CSE  

---

## 1. Demonstration Timing & Outline (5–8 Minutes Total)

| Time | Demonstration Phase | Key Talking Points & Actions |
| :---: | :--- | :--- |
| **0:00 – 1:00** | **1. Motivation & Problem Statement** | Introduce the risks of photorealistic synthetic media (Stable Diffusion), lack of visible artifacts to the naked human eye, and the objective: binary authenticity detection ($P(\text{REAL})$ vs. $P(\text{FAKE})$). |
| **1:00 – 2:00** | **2. Dataset & Preprocessing Pipeline** | Explain the **CIFAKE** benchmark (120,000 images, 32×32 RGB, balanced 50/50). Highlight the strict split: 80k train, 20k validation, 20k held-out test. Show how internal `Rescaling` maps pixels to $[-1, 1]$ without double-normalization bugs. |
| **2:00 – 3:30** | **3. MobileNetV2 Transfer Learning Architecture** | Explain why MobileNetV2 was chosen: inverted residual bottlenecks, depthwise separable convolutions, lightweight footprint (~2.2M params), frozen ImageNet backbone, Global Average Pooling, Dropout (0.2), and single sigmoid neuron. |
| **3:30 – 5:00** | **4. Google Colab Execution & Training Curves** | Walk through [`project_colab_fixed.ipynb`](file:///c:/Users/sreya/OneDrive/Documents/Deep_Learning_Project/branch2/DL-Mini-Project-AI-Image-Detection/project_colab_fixed.ipynb). Show GPU detection, automated `kagglehub` download, and loss/accuracy convergence curves across 5 epochs. Discuss how `BinaryAccuracy(threshold=0.5)` solved the 0.00% accuracy phenomenon. |
| **5:00 – 6:30** | **5. Live Web App Demonstration (`app.py`)** | Launch `streamlit run app.py`. Test live inference on an authentic photo (`REAL`) and an AI-generated rendering (`FAKE`). Show real-time probability scores, confidence gauge, and the dynamic threshold slider. |
| **6:30 – 7:30** | **6. Test Evaluation & Confusion Matrix** | Present the 20,000 held-out test set metrics (~93%+ accuracy, balanced precision/recall, and the confusion matrix showing minimal false positives). |
| **7:30 – 8:00** | **7. Limitations, Future Scope & Q&A** | Summarize limitations (Stable Diffusion v1.4 specificity, 160×160 resolution), future work (multi-generator ensembles, Grad-CAM explainability), and invite faculty questions. |

---

## 2. Step-by-Step Live Demonstration Instructions

### Step 1: Launch the Interactive Web Application
Open your terminal (PowerShell or Command Prompt) in the project workspace:
```powershell
cd c:\Users\sreya\OneDrive\Documents\Deep_Learning_Project\branch2\DL-Mini-Project-AI-Image-Detection
streamlit run app.py
```
- The browser will open automatically at: `http://localhost:8501`.
- Explain the sidebar: University branding, student roll numbers, guide details, and the **Decision Threshold Slider** (default: `0.50`).

### Step 2: Test an Authentic Photographic Capture (Ground Truth: REAL)
1. In the Web UI, select **"Choose from CIFAKE Test Samples"** or upload a photo from:
   ```text
   CSE0243-0264-0275/sample_dataset/REAL/0000 (10).jpg
   ```
2. Point out:
   - Green confirmation banner: **"PREDICTION: REAL (Authentic Photo)"**
   - High confidence score ($P(\text{REAL}) \ge 85\%$).
   - The probability breakdown chart indicating near-zero synthetic probability.

### Step 3: Test an AI-Generated Synthetic Rendering (Ground Truth: FAKE)
1. Switch to:
   ```text
   CSE0243-0264-0275/sample_dataset/FAKE/0 (10).jpg
   ```
2. Point out:
   - Red alert banner: **"PREDICTION: FAKE (AI-Generated Image)"**
   - High synthetic confidence ($P(\text{REAL}) \le 20\%$, meaning $\ge 80\%$ FAKE confidence).
   - Explain that the convolutional feature extractor detected characteristic diffusion frequency patterns in pixel transitions.

### Step 4: Show the Evaluation Metrics Dashboard Tab
1. Click the **"Model Evaluation & Metrics"** tab in the web app.
2. Show the evaluator the four KPI metric cards:
   - **Test Accuracy**: ~93.93%
   - **Macro Precision**: ~93.94%
   - **Macro Recall**: ~93.93%
   - **Macro F1-Score**: ~93.93%
3. Show the embedded high-resolution **Confusion Matrix Heatmap** and **Loss/Accuracy Curves**.

---

## 3. Contingency & Backup Plan (If Live Application or Internet Fails)

### Scenario A: Web App Port or Browser Hangs
If Streamlit is blocked by firewall or port conflicts:
- Run the single-image CLI command directly in the terminal:
  ```powershell
  python predict.py --image "CSE0243-0264-0275/sample_dataset/REAL/0000 (10).jpg"
  python predict.py --image "CSE0243-0264-0275/sample_dataset/FAKE/0 (10).jpg"
  ```
- This prints the formatted ASCII inference table in less than 2 seconds without requiring a web browser.

### Scenario B: No Internet Connectivity During Presentation
- The offline sample dataset is already locally available in:
  `CSE0243-0264-0275/sample_dataset/`
- The model checkpoint `models/mobilenetv2_cifake_best.keras` is stored locally.
- The web app and CLI script run **100% locally** without needing an active internet connection.

### Scenario C: Evaluator Requests Proof of Code Correctness
- Run the automated pytest suite live in terminal:
  ```powershell
  python -m pytest tests/
  ```
- All **12 / 12 unit tests pass in under 60 seconds**, proving directory validation, mapping integrity, layer shapes, and metric formulas.
