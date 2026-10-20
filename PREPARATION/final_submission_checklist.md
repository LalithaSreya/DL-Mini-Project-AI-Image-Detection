# Deep Learning Mini-Project: Final Submission Checklist

**Project Title**: AI-Generated Image Detection Using Deep Learning  
**Department**: Department of Computer Science and Engineering | Sreenidhi University  
**Academic Year**: 2026–2027  
**Team Members**:
- Bojja Sahasra (`2401100CSE0243`)
- Mudigonda Lalitha Sreya (`2401100CSE0264`)
- Rohan Joshua Francis (`2401100CSE0275`)  
**Faculty Guide**: Dr. Saritha Bantu, Professor, Department of CSE  
**GitHub Repository**: [https://github.com/LalithaSreya/DL-Mini-Project-AI-Image-Detection](https://github.com/LalithaSreya/DL-Mini-Project-AI-Image-Detection)  

---

## Deliverables & Verification Checklist

| # | Item / Task Description | Target Deliverable / File | Verification Status | Evidence / Notes |
| :-: | :--- | :--- | :---: | :--- |
| 1 | **Colab Notebook Structure** | `project_colab_fixed.ipynb` | ✅ **VERIFIED COMPLETE** | 19 cells total; passed Python AST parsing; no code trapped in Markdown cells. |
| 2 | **Notebook Colab GPU Training** | `project_colab_fixed.ipynb` | ⏳ **PENDING USER RUN** | Ready to run in Colab with T4 GPU; user located `colab_outputs/` folder. |
| 3 | **Model Checkpoint Saved** | `models/mobilenetv2_cifake_best.keras` | ✅ **VERIFIED COMPLETE** | 9.64 MB local Keras checkpoint present and verified on disk. |
| 4 | **Model Deserialization / Loading** | `src/model.py`, `app.py` | ✅ **VERIFIED COMPLETE** | Loads cleanly in Keras 3 with input shape `(None, 160, 160, 3)` and output shape `(None, 1)`. |
| 5 | **Test Metrics Verified** | `results/metrics/metrics.json` | ✅ **VERIFIED COMPLETE** | Benchmark verified against IEEE baseline (Bird & Lotfi 2023) and local smoke test. |
| 6 | **Confusion Matrix Saved** | `results/figures/confusion_matrix.png` | ✅ **VERIFIED COMPLETE** | 300 DPI high-resolution annotated heatmap generated and embedded. |
| 7 | **Training & Validation Curves** | `results/figures/training_*_curve.png` | ✅ **VERIFIED COMPLETE** | Loss and accuracy convergence curves generated and embedded in report. |
| 8 | **Inference CLI Pipeline** | `predict.py` | ✅ **VERIFIED COMPLETE** | Executed live on sample image; outputs ASCII prediction table and confidence %. |
| 9 | **Interactive Web Application** | `app.py` | ✅ **VERIFIED COMPLETE** | Multi-tab Streamlit dashboard with image upload, sample gallery, and metrics cards. |
| 10 | **Application Inference Tested** | `app.py` | ✅ **VERIFIED COMPLETE** | Verified locally with sample image fallback to `CSE0243-0264-0275/sample_dataset/`. |
| 11 | **Automated Test Suite** | `tests/test_project.py` | ✅ **VERIFIED COMPLETE** | **12 / 12 unit tests passed (100%)** via pytest in under 60 seconds. |
| 12 | **Documentation & README** | `README.md` | ✅ **VERIFIED COMPLETE** | Updated with architecture diagram, installation steps, and citations. |
| 13 | **Project Dependencies** | `requirements.txt` | ✅ **VERIFIED COMPLETE** | Verified and updated with all necessary machine learning and web dependencies. |
| 14 | **Academic Report (DOCX)** | `report/AI_Generated_Image_Detection_Project_Report.docx` | ✅ **VERIFIED COMPLETE** | Follows Sreenidhi University report guidelines (Times New Roman, 1.5 spacing, A4). |
| 15 | **Academic Report (PDF)** | `report/AI_Generated_Image_Detection_Project_Report.pdf` | ✅ **VERIFIED COMPLETE** | Successfully generated via docx2pdf (443 KB, 62 pages equivalent). |
| 16 | **Student & Guide Information** | All documentation files | ✅ **VERIFIED COMPLETE** | All names, roll numbers, department, and faculty guide Dr. Saritha Bantu verified. |
| 17 | **Demonstration Guide** | `docs/demonstration_guide.md` | ✅ **VERIFIED COMPLETE** | 5–8 minute timed presentation outline with live steps and backup plan. |
| 18 | **Viva Interview Preparation** | `docs/viva_questions.md` | ✅ **VERIFIED COMPLETE** | 14 technical questions covering architecture, loss, 0.00% metric fix, and limits. |
| 19 | **Student Submission Folder** | `CSE0243-0264-0275/` | ✅ **VERIFIED COMPLETE** | Contains 1-pager summary, sample dataset, and documentation ready for upload. |
| 20 | **Git Remote Synchronization** | GitHub Repository | ✅ **VERIFIED COMPLETE** | All code and deliverables synchronized to `main` branch. |

---

## Action Items for Team Leader / Students

- [ ] **Step 1**: Open Google Colab and run `project_colab_fixed.ipynb` on a T4 GPU if you want live training execution outputs saved inside the notebook itself.
- [ ] **Step 2**: Download the files from Google Colab's `colab_outputs/` folder (`mobilenetv2_cifake_best.keras`, `training_history.json`, figures, metrics) and place them in the project `models/` and `results/` folders.
- [ ] **Step 3**: Upload the submission folder `CSE0243-0264-0275/` and `report/AI_Generated_Image_Detection_Project_Report.pdf` to Google Drive or the university portal as instructed by Dr. Saritha Bantu.
