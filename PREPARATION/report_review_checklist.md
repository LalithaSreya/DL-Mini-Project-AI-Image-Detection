# Academic Report Review & Verification Checklist
## AI-Generated Image Detection Using Deep Learning
**Academic Year**: 2026–2027 | **Institution**: Sreenidhi University | **Department**: Computer Science and Engineering

> **IMPORTANT NOTICE**:
> This checklist is a personal quality-assurance tool for team members before submitting the academic report to Dr. Saritha Bantu and the university examination panel.
>
> In accordance with academic audit standards, items requiring human inspection, faculty guide signoff, or physical signatures are marked **PENDING MANUAL VERIFICATION**.

---

## 1. Institutional & Administrative Information

| Check Item | Requirement Description | Status | Verification Notes |
|:---|:---|:---:|:---|
| **University Name** | Exact name: "SREENIDHI UNIVERSITY" |  VERIFIED | Verified on Cover Page, Certificate, Declaration, and Headers in both DOCX and PDF. |
| **Department Name** | "Department of Computer Science and Engineering" |  VERIFIED | Confirmed across preliminary pages and report chapter headers. |
| **Academic Year** | "2026–2027" |  VERIFIED | Updated across DOCX, PDF, README, and submission files. |
| **Project Title** | "AI-Generated Image Detection Using Deep Learning" |  VERIFIED | Consistent across report, notebook, and web application. |
| **Student Details** | - Bojja Sahasra (`2401100CSE0243`)<br>- Mudigonda Lalitha Sreya (`2401100CSE0264`)<br>- Rohan Joshua Francis (`2401100CSE0275`) |  VERIFIED | Roll numbers and student names verified against registered team records. |
| **Project Guide Details** | Dr. Saritha Bantu, Professor, Department of CSE (`saritha.b@suh.edu.in`) |  VERIFIED | Formatted on Certificate and Acknowledgements. |
| **Guide Physical Signature** | Physical signature on Certificate page | **PENDING** | Must be physically signed by Dr. Saritha Bantu prior to final hard-copy binding. |
| **Student Signatures** | Signatures on Declaration page | **PENDING** | All three team members must physically sign the Declaration page in the printed report. |
| **Submission Format Rules**| Check whether the department requires hard-bound book, spiral, or LMS PDF upload | **PENDING** | Needs explicit confirmation from university exam coordinator or Dr. Saritha Bantu. |

---

## 2. Formatting & University Template Conformance

| Check Item | Requirement Description | Status | Verification Notes |
|:---|:---|:---:|:---|
| **Font Style** | Times New Roman throughout entire document |  VERIFIED | Document styles strictly configured to Times New Roman. |
| **Font Sizes** | Title: 16pt Bold, Chapter: 14–16pt Bold, Section: 12–14pt Bold, Body: 12pt Regular |  VERIFIED | Conforms to `Deep_Learning_Project_Report_Template_TOC.docx` hierarchy. |
| **Line Spacing** | 1.5 line spacing for body paragraphs |  VERIFIED | Verified across all 7 chapters. |
| **Page Margins** | Left: 1.5 inches (binding margin); Top, Right, Bottom: 1.0 inch |  VERIFIED | Page setup sections configured with left margin = $1.5''$ and others = $1.0''$. |
| **Paper Size** | Standard A4 ($210 \times 297\text{ mm}$) |  VERIFIED | Confirmed in DOCX section page setup. |
| **Preliminary Pages** | Certificate, Declaration, Acknowledgements, Abstract, Table of Contents, Lists of Figures/Tables/Abbreviations |  VERIFIED | Formatted with Roman numerals (i, ii, iii...); Chapters start with Arabic numeral 1. |
| **Table of Contents** | Complete hierarchy matching Chapter 1 to 7 and Appendices |  VERIFIED | Verified; page numbers match headings. |
| **DOCX vs PDF Parity** | Formatting, alignment, and page breaks identical in DOCX and PDF |  VERIFIED | Converted using `docx2pdf`; confirmed visual alignment. |

---

## 3. Abstract, Objectives & Literature Review

| Check Item | Requirement Description | Status | Verification Notes |
|:---|:---|:---:|:---|
| **Abstract Content** | Concisely states problem, methodology, benchmark dataset, and results |  VERIFIED | Under 300 words, clearly states MobileNetV2 transfer learning on CIFAKE. |
| **Objectives** | Explicit, measurable objectives defined in Chapter 1 |  VERIFIED | Ingestion, transfer learning, reproducible pipeline, verification, and UI. |
| **Problem Formulation** | Mathematical formulation of binary classification ($y \in \{0, 1\}$) |  VERIFIED | Defined in Section 1.3 and Section 3.1. |
| **Literature References** | Minimum 5 academic papers cited in IEEE style |  VERIFIED | 6 foundational papers cited (Bird & Lotfi 2023, Sandler et al. 2018, Rombach et al. 2022, Wang et al. 2020, Krizhevsky 2009, Goodfellow et al. 2014). |
| **Citation Cross-Checks** | Every `[1]`, `[2]`, `[3]` in text matches Reference entry |  VERIFIED | Citations cross-referenced in Chapter 2, 3, and References. |

---

## 4. Dataset, Preprocessing & Architecture Verification

| Check Item | Requirement Description | Status | Verification Notes |
|:---|:---|:---:|:---|
| **Dataset Source & URL** | CIFAKE Kaggle dataset source cited |  VERIFIED | Cited Kaggle URL (`birdy654/cifake-real-and-ai-generated-synthetic-images`). |
| **Dataset Counts** | 120,000 total images: 100k train (50k fake, 50k real), 20k test (10k fake, 10k real) |  VERIFIED | Exact numbers stated in Section 3.2. |
| **Input Dimensions** | $160 \times 160 \times 3$ RGB |  VERIFIED | Matching `src/model.py`, `app.py`, and `project_colab_fixed.ipynb`. |
| **In-Graph Preprocessing**| Rescaling to $[-1, 1]$ via `scale=1/127.5, offset=-1.0` |  VERIFIED | Verified in model architecture definition. |
| **Class Label Mapping** | FAKE = 0, REAL = 1 |  VERIFIED | Verified across all scripts, notebook, report, and Streamlit app. |
| **Backbone & Freezing** | MobileNetV2 with ImageNet weights, `trainable = False` |  VERIFIED | Confirmed frozen feature extractor. |
| **Loss & Optimizer** | Binary Cross-Entropy with Adam ($\text{learning\_rate} = 10^{-4}$) |  VERIFIED | Verified in Section 3.4. |

---

## 5. Metrics, Experimental Results & Diagnostic Transparency

| Check Item | Requirement Description | Status | Verification Notes |
|:---|:---|:---:|:---|
| **Distinction: Subset vs Full Training** | Clearly distinguishes local 1-epoch smoke test from 5-epoch full Colab benchmark |  VERIFIED | Explicitly documented in Section 5.1 and Section 5.2. No fabricated 99% claims. |
| **The 0.00% Accuracy Phenomenon** | Full technical explanation of `tf.keras.metrics.Accuracy()` vs `BinaryAccuracy(threshold=0.5)` |  VERIFIED | Documented in Section 5.3 and Appendix B. Prevents panel confusion. |
| **Confusion Matrix** | Visual confusion matrix graphic included and discussed |  VERIFIED | Rendered in Figure 5.3 and saved in `results/figures/confusion_matrix.png`. |
| **Loss & Accuracy Trajectories** | Training vs. validation curve figures included |  VERIFIED | Figures 5.1 and 5.2 present in `results/figures/`. |
| **Full Colab GPU Retraining (Optional)** | Full 5 epochs run on Colab to replace subset checkpoint weights | **PENDING** | Student can run `project_colab_fixed.ipynb` on Colab GPU if they wish to include full-run weights. |

---

## 6. Software, Interfaces & Deliverables

| Check Item | Requirement Description | Status | Verification Notes |
|:---|:---|:---:|:---|
| **Executable Notebook** | `project_colab_fixed.ipynb` runs top-to-bottom without manual intervention |  VERIFIED | Tested with automated `kagglehub` download and dynamic path resolution. |
| **Streamlit Web Application** | `app.py` runs with image upload and sample image presets |  VERIFIED | Tested with local sample dataset fallback; path resolution verified. |
| **Requirements Specification** | `requirements.txt` has pinned/minimum versions |  VERIFIED | Includes TensorFlow, Streamlit, Scikit-learn, Matplotlib, Kagglehub, Docx2pdf. |
| **Git Repository Consistency** | Clean branch, descriptive commit history, public accessibility |  VERIFIED | Branch `main`, commit `2ea221f`, pushed to GitHub repository. |
| **Demonstration Guide** | Step-by-step 5–8 minute presentation walkthrough |  VERIFIED | Documented in `SUBMISSION/docs/demonstration_guide.md`. |
| **Technical Viva Questions** | Answers prepared for common panel questions |  VERIFIED | 14 questions prepared in `PREPARATION/viva_questions.md`. |

---

## 7. Final Student Action Items Before Viva

- [ ] **Action 1**: Open [AI_Generated_Image_Detection_Project_Report.docx](file:///c:/Users/sreya/OneDrive/Documents/Deep_Learning_Project/branch2/DL-Mini-Project-AI-Image-Detection/report/AI_Generated_Image_Detection_Project_Report.docx) and conduct a final manual read-through.
- [ ] **Action 2**: Print the report (or present the PDF) to Dr. Saritha Bantu for guide review and signature.
- [ ] **Action 3**: Ensure all three team members (Sahasra, Lalitha Sreya, Rohan Joshua) sign the Declaration page.
- [ ] **Action 4**: Confirm whether your department requires a spiral-bound or hard-bound copy for submission.
- [ ] **Action 5**: Review [viva_questions.md](file:///c:/Users/sreya/OneDrive/Documents/Deep_Learning_Project/branch2/DL-Mini-Project-AI-Image-Detection/PREPARATION/viva_questions.md) and practice the 2-minute elevator pitch.
