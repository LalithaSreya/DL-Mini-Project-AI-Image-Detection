"""
AI-Generated Image Detection - Interactive Streamlit Web Interface
SREENIDHI UNIVERSITY - Department of Computer Science and Engineering

Team Members:
- Bojja Sahasra (2401100CSE0243)
- Rohan Joshua Francis (2401100CSE0275)
- Mudigonda Lalitha Sreya (2401100CSE0264)
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
import tensorflow as tf

from src.dataset import CLASS_NAMES, LABEL_TO_CLASS


# Page configuration
st.set_page_config(
    page_title="AI Image Detector | Sreenidhi University",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .real-card {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .fake-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained MobileNetV2 detector...")
def load_detector_model(model_path: str = "models/mobilenetv2_cifake_best.keras") -> Optional[tf.keras.Model]:
    """Loads and caches the trained Keras model."""
    path = Path(model_path)
    if not path.is_file():
        # Fallback to final model if best is not present
        alt_path = Path("models/mobilenetv2_cifake_final.keras")
        if alt_path.is_file():
            path = alt_path
        else:
            return None
    return tf.keras.models.load_model(str(path))


def preprocess_pil_image(image: Image.Image, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """Preprocesses a PIL image for model inference."""
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_image, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def run_inference(model: tf.keras.Model, image: Image.Image, threshold: float = 0.5):
    """Executes prediction on the input image."""
    tensor = preprocess_pil_image(image)
    raw_score = float(model.predict(tensor, verbose=0)[0][0])
    
    is_real = raw_score >= threshold
    predicted_class = "REAL" if is_real else "FAKE"
    
    prob_real = raw_score
    prob_fake = 1.0 - raw_score
    
    confidence = (prob_real if is_real else prob_fake) * 100.0
    return predicted_class, raw_score, confidence, prob_real, prob_fake


def get_sample_images():
    """Fetches preset sample images from the CIFAKE test folder if available."""
    samples = {}
    real_dir = Path("data/CIFAKE/test/REAL")
    fake_dir = Path("data/CIFAKE/test/FAKE")

    if real_dir.is_dir():
        real_files = sorted(list(real_dir.glob("*.jpg")))[:3]
        for idx, f in enumerate(real_files, 1):
            samples[f"Sample Real Image #{idx}"] = f

    if fake_dir.is_dir():
        fake_files = sorted(list(fake_dir.glob("*.jpg")))[:3]
        for idx, f in enumerate(fake_files, 1):
            samples[f"Sample AI-Generated Image #{idx}"] = f

    return samples


def main():
    # Sidebar: University & Team Info
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
        st.markdown("### **SREENIDHI UNIVERSITY**")
        st.caption("Department of Computer Science and Engineering")
        st.markdown("**Academic Year**: 2025–2026")
        st.markdown("---")
        
        st.markdown("#### **Project Team Members**")
        st.markdown(
            """
            - **Bojja Sahasra**  
              `2401100CSE0243`
            - **Rohan Joshua Francis**  
              `2401100CSE0275`
            - **Mudigonda Lalitha Sreya**  
              `2401100CSE0264`
            """
        )
        st.markdown("---")
        st.markdown("#### **Course Coordinator**")
        st.markdown("**Dr. Saritha Bantu**  \nProfessor, Dept. of CSE, Sreenidhi University")
        st.markdown("---")
        
        st.markdown("#### **Detection Settings**")
        threshold = st.slider(
            "Decision Threshold (P(REAL))",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Images with P(REAL) >= threshold are classified as REAL.",
        )
        st.caption("Default: 0.50 (Standard binary decision boundary)")

    # Main Header
    st.markdown('<div class="main-title">AI-Generated Image Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Deep Learning Binary Classifier powered by Transfer Learning with <b>MobileNetV2</b> on the <b>CIFAKE</b> Benchmark</div>',
        unsafe_allow_html=True,
    )

    # Navigation tabs
    tab_detector, tab_metrics, tab_about = st.tabs(["🔍 Live Image Detector", "📊 Model Evaluation & Metrics", "📖 About the Project"])

    # Load Model
    model = load_detector_model()
    if model is None:
        st.error(
            "⚠️ No trained model checkpoint found in `models/`!\n"
            "Please train the model first by running `python main.py` in your terminal."
        )
        return

    # ---------------- TAB 1: Live Image Detector ----------------
    with tab_detector:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("1. Select or Upload Image")
            input_mode = st.radio("Choose Input Mode:", ["Upload Image File", "Choose from CIFAKE Test Samples"], horizontal=True)

            selected_image = None
            image_name = ""

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Upload a photo (.jpg, .jpeg, .png)",
                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                    help="Upload any natural photograph or synthetic AI rendering.",
                )
                if uploaded_file is not None:
                    try:
                        selected_image = Image.open(uploaded_file)
                        image_name = uploaded_file.name
                    except Exception as e:
                        st.error(f"Failed to open image: {e}")
            else:
                sample_dict = get_sample_images()
                if not sample_dict:
                    st.info("CIFAKE test folder not found. Please upload an image instead.")
                else:
                    sample_choice = st.selectbox("Select a benchmark sample:", list(sample_dict.keys()))
                    if sample_choice:
                        sample_path = sample_dict[sample_choice]
                        selected_image = Image.open(sample_path)
                        image_name = sample_path.name

            if selected_image is not None:
                st.image(selected_image, caption=f"Selected: {image_name}", use_container_width=True)

        with col_result:
            st.subheader("2. Detection Analysis")
            if selected_image is None:
                st.info("👈 Please upload an image or choose a sample from the left panel to run analysis.")
            else:
                with st.spinner("Analyzing image features..."):
                    pred_class, raw_score, confidence, prob_real, prob_fake = run_inference(
                        model, selected_image, threshold=threshold
                    )

                # Prominent Result Card
                if pred_class == "REAL":
                    st.markdown(
                        f"""
                        <div class="prediction-card real-card">
                            <h2 style="margin:0; color:#065F46;">🟢 AUTHENTIC PHOTOGRAPH (REAL)</h2>
                            <p style="margin-top:0.5rem; font-size:1.1rem;">
                                The model classifies this image as a <b>Natural Photograph</b> with <b>{confidence:.2f}%</b> confidence.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-card fake-card">
                            <h2 style="margin:0; color:#991B1B;">🔴 AI-GENERATED SYNTHETIC (FAKE)</h2>
                            <p style="margin-top:0.5rem; font-size:1.1rem;">
                                The model classifies this image as an <b>AI-Generated Synthetic Image</b> with <b>{confidence:.2f}%</b> confidence.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Probability Breakdowns
                st.markdown("#### Probability Distribution")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("P(REAL) - Authentic", f"{prob_real * 100:.2f}%")
                    st.progress(float(prob_real))
                with col_p2:
                    st.metric("P(FAKE) - AI-Generated", f"{prob_fake * 100:.2f}%")
                    st.progress(float(prob_fake))

                # Technical Inspection
                with st.expander("🛠️ View Detailed Technical Breakdown"):
                    st.write(
                        {
                            "Raw Model Output (Sigmoid)": f"{raw_score:.6f}",
                            "Applied Decision Threshold": f"{threshold:.2f}",
                            "Input Resolution Rescaling": "160 x 160 x 3",
                            "Feature Extractor": "MobileNetV2 (ImageNet weights, Frozen)",
                            "Classification Head": "GlobalAvgPool2D -> Dropout(0.2) -> Dense(1, Sigmoid)",
                        }
                    )

    # ---------------- TAB 2: Model Evaluation & Metrics ----------------
    with tab_metrics:
        st.subheader("Model Performance & Academic Benchmark Metrics")
        st.write("Quantitative results generated from held-out evaluation on the CIFAKE dataset:")

        # Summary KPIs
        metrics_file = Path("results/metrics.json")
        csv_file = Path("results/metrics.csv")

        if metrics_file.is_file():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Test Accuracy", f"{metrics_data.get('accuracy', 0.0) * 100:.2f}%")
            with col_m2:
                st.metric("Macro Precision", f"{metrics_data.get('macro_avg', {}).get('precision', 0.0) * 100:.2f}%")
            with col_m3:
                st.metric("Macro Recall", f"{metrics_data.get('macro_avg', {}).get('recall', 0.0) * 100:.2f}%")
            with col_m4:
                st.metric("Macro F1-Score", f"{metrics_data.get('macro_avg', {}).get('f1_score', 0.0) * 100:.2f}%")

        if csv_file.is_file():
            st.markdown("#### Detailed Classification Breakdown")
            df = pd.read_csv(csv_file)
            st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Visual Evaluation Graphs")
        col_g1, col_g2 = st.columns(2)

        cm_img = Path("results/confusion_matrix.png")
        loss_img = Path("results/training_loss_curve.png")
        acc_img = Path("results/training_accuracy_curve.png")

        with col_g1:
            if cm_img.is_file():
                st.image(str(cm_img), caption="Test Split Confusion Matrix", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found. Run `python main.py` to generate it.")

        with col_g2:
            if loss_img.is_file():
                st.image(str(loss_img), caption="Training vs. Validation Loss Curve", use_container_width=True)
            if acc_img.is_file():
                st.image(str(acc_img), caption="Training vs. Validation Accuracy Curve", use_container_width=True)

    # ---------------- TAB 3: About the Project ----------------
    with tab_about:
        st.subheader("Project Architecture & Academic Background")
        st.markdown(
            """
            ### Problem Statement
            With the rapid proliferation of Latent Diffusion Models (e.g. Stable Diffusion) and Generative Adversarial Networks (GANs),
            synthesizing photorealistic media has become commonplace. Distinguishing authentic visual evidence from synthetically fabricated
            imagery is vital for mitigating online disinformation, identity impersonation, and fraud.

            ### Methodology: Transfer Learning with MobileNetV2
            1. **Preprocessing Layer**: Built-in layer that scales raw pixel values $[0, 255]$ into $[-1, 1]$.
            2. **Frozen Feature Extractor**: Base MobileNetV2 pretrained on ImageNet (1.4 million images, 1,000 classes), preserving general low-level edge and texture representations.
            3. **Global Average Pooling**: Condenses spatial feature representations $(5 \\times 5 \\times 1280)$ into a 1280-dimensional feature vector.
            4. **Dropout Regularization ($p = 0.2$)**: Alleviates co-adaptation and prevents overfitting.
            5. **Sigmoid Classification Output**: Computes probability $P(\\text{REAL}) \\in [0, 1]$.
               - Score $\\ge 0.50 \\implies$ **REAL** (Natural Photograph)
               - Score $< 0.50 \\implies$ **FAKE** (AI-Generated Synthetic Image)

            ### Benchmark Dataset
            - **CIFAKE Benchmark**: 120,000 total images evenly divided between authentic CIFAR-10 images and Stable Diffusion v1.4 renderings across 10 categories.
            """
        )


if __name__ == "__main__":
    main()
