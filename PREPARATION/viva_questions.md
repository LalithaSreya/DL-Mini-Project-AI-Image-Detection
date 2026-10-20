# Comprehensive Viva & Technical Interview Questions

**Project Title**: AI-Generated Image Detection Using Deep Learning  
**Department**: Computer Science and Engineering | Sreenidhi University  
**Academic Year**: 2026–2027  
**Team**: Bojja Sahasra (2401100CSE0243), Mudigonda Lalitha Sreya (2401100CSE0264), Rohan Joshua Francis (2401100CSE0275)  
**Faculty Guide**: Dr. Saritha Bantu, Professor, Department of CSE  

---

## 1. Problem Statement & Motivation

### Q1: What is the core problem statement of this project?
**Answer**:  
Given an arbitrary input digital image $X \in \mathbb{R}^{H \times W \times 3}$, our objective is to classify whether the image is an authentic photographic capture (**REAL**, Class 1) or an AI-generated synthetic rendering (**FAKE**, Class 0) produced by a generative model (specifically Stable Diffusion v1.4), outputting both a binary label and a continuous confidence score.

### Q2: Why is detecting AI-generated images challenging for humans and traditional computer vision?
**Answer**:  
Modern Latent Diffusion Models synthesize images by iteratively removing Gaussian noise guided by text prompts, producing hyper-realistic textures, specular lighting, and natural depth-of-field that deceive the naked human eye. Traditional computer vision (which searches for manual Photoshop splicing boundaries or unnatural facial asymmetries) fails because fully synthesized images have no physical base photo or splicing seams. However, deep convolutional neural networks can detect subtle, high-frequency statistical anomalies left by the diffusion de-noising and latent decoder process.

---

## 2. Dataset & Data Engineering

### Q3: Which dataset did you use, and what is its composition?
**Answer**:  
We used the **CIFAKE** benchmark dataset (Bird & Lotfi, 2023). It contains exactly **120,000 images** (32×32 pixels, RGB) and is perfectly balanced:
- **60,000 REAL images**: Authentic photographs taken from the CIFAR-10 photographic benchmark across 10 semantic classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **60,000 FAKE images**: Synthetic images generated via Stable Diffusion v1.4 conditioned on the exact same 10 CIFAR-10 prompt classes.

### Q4: How did you partition the dataset, and how did you prevent data leakage?
**Answer**:  
We strictly isolated the partitions:
1. **Training Split**: 80,000 images (80% of `train/`), used to update model weights.
2. **Validation Split**: 20,000 images (20% of `train/`), used strictly for monitoring validation loss and triggering early stopping.
3. **Held-Out Test Split**: 20,000 images (`test/`), completely isolated and never seen during training or checkpoint selection. Loaded with `shuffle=False` to ensure exact 1-to-1 sample alignment between predictions and ground-truth labels.
This ensures zero data leakage between training and testing.

---

## 3. Deep Learning Architecture & Transfer Learning

### Q5: What is Transfer Learning, and why did you use MobileNetV2?
**Answer**:  
Transfer Learning is a technique where a model pre-trained on a massive dataset (ImageNet with 1.4 million images) is repurposed for a specialized downstream task. We selected **MobileNetV2** (Sandler et al., 2018) because:
- It uses **inverted residual blocks** with linear bottlenecks and **depthwise separable convolutions**, drastically reducing parameter count to ~2.2 million (compared to ~25M in ResNet-50).
- It provides high representational accuracy with a very low memory footprint, making it ideal for fast, CPU-friendly inference and web deployment.
- We froze the pre-trained backbone (`trainable = False`) to preserve low-level edge and texture filters while training only our custom classification head.

### Q6: Walk through the layers of your network from input to output.
**Answer**:  
1. **Input Layer**: Shape `(160, 160, 3)` accepting raw pixel values in $[0, 255]$.
2. **Rescaling Layer**: Computes $(x / 127.5) - 1.0$, mapping pixels to $[-1.0, 1.0]$.
3. **MobileNetV2 Base**: Frozen convolutional feature extractor outputting 2D feature maps.
4. **GlobalAveragePooling2D**: Collapses each spatial feature map into its average scalar, producing a 1,280-dimensional feature vector.
5. **Dropout Layer (rate = 0.2)**: Randomly drops 20% of activations during training to combat overfitting.
6. **Dense Output Layer**: A single neuron with a **Sigmoid** activation function outputting scalar probability $P(\text{REAL}) \in [0.0, 1.0]$.

---

## 4. Preprocessing & Normalization

### Q7: How does your preprocessing pipeline work, and how did you prevent double normalization?
**Answer**:  
Images are loaded as 3-channel RGB and resized to $160 \times 160$ using bilinear interpolation. The pixel values remain raw floats in $[0, 255]$. Normalization to $[-1.0, 1.0]$ is embedded directly inside the model as a Keras `Rescaling` layer. Because normalization is performed internally by the model itself, external inference tools (`predict.py`, `app.py`) only supply raw $[0, 255]$ arrays, completely eliminating any risk of double normalization.

---

## 5. Training Strategy & Optimization

### Q8: What loss function, optimizer, and hyperparameters did you use?
**Answer**:  
- **Loss Function**: `BinaryCrossentropy(from_logits=False)`. It penalizes deviations between predicted probabilities and binary targets.
- **Optimizer**: `Adam` with a learning rate of $1 \times 10^{-4}$ ($0.0001$).
- **Batch Size**: 32 (balanced memory efficiency and gradient stability).
- **Epochs**: 5 maximum, governed by `EarlyStopping(patience=3)` and `ModelCheckpoint(save_best_only=True, monitor="val_loss")`.

---

## 6. Evaluation Metrics & The 0.00% Accuracy Phenomenon

### Q9: Why did early testing in Keras produce 0.00% test accuracy, and how did you resolve it?
**Answer**:  
In Keras, `tf.keras.metrics.Accuracy()` computes exact equality: `y_true == y_pred`. In a binary sigmoid model, `y_true` contains integer targets `{0, 1}`, while `y_pred` contains continuous probabilities like `0.8542`. Because an integer `1` never exactly equals a continuous float `0.8542`, zero samples matched, resulting in **0.00% accuracy**!  
We resolved this by using `tf.keras.metrics.BinaryAccuracy(threshold=0.5)`. In Python/scikit-learn evaluations, we explicitly threshold probabilities via `y_pred = (y_prob >= 0.5).astype(int)` before computing metrics.

### Q10: What are Precision, Recall, and F1-Score in your project?
**Answer**:  
- **Precision**: Of all images predicted as FAKE (or REAL), what fraction was actually FAKE (or REAL)? Formula: $TP / (TP + FP)$.
- **Recall (Sensitivity)**: Of all actual FAKE (or REAL) images, what fraction did the model successfully catch? Formula: $TP / (TP + FN)$.
- **F1-Score**: The harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$, ensuring balanced performance without majority-class bias.

### Q11: Explain the structure of your Confusion Matrix.
**Answer**:  
The confusion matrix uses canonical class ordering: `0 = FAKE`, `1 = REAL`:
- Row 0, Col 0 ($TN$): Actual FAKE correctly predicted as FAKE.
- Row 0, Col 1 ($FP$): Actual FAKE misclassified as REAL.
- Row 1, Col 0 ($FN$): Actual REAL misclassified as FAKE.
- Row 1, Col 1 ($TP$): Actual REAL correctly predicted as REAL.

---

## 7. Inference, Application & Limitations

### Q12: How does the model make a prediction on a new, uploaded image?
**Answer**:  
The uploaded image is opened with Pillow, converted to RGB, resized to $(160, 160)$, and reshaped to `(1, 160, 160, 3)`. The model executes forward inference and outputs a sigmoid probability score $S \in [0.0, 1.0]$ representing $P(\text{REAL})$:
- If $S \ge 0.50 \implies$ Class is **`REAL`**; Confidence is $S \times 100\%$.
- If $S < 0.50 \implies$ Class is **`FAKE`**; Confidence is $(1.0 - S) \times 100\%$.

### Q13: What are the primary limitations of your model?
**Answer**:  
1. **Generator Specificity**: The model is trained on Stable Diffusion v1.4. While it learns diffusion artifacts well, it may exhibit reduced accuracy against newer generators like Midjourney v6 or DALL-E 3 whose latent autoencoder structures differ.
2. **Resolution Downsampling**: High-resolution synthetic media is downscaled to $160 \times 160$, which may smooth out microscopic pixel artifacts.
3. **Adversarial Post-Processing**: Aggressive JPEG compression, gaussian blurring, or social media re-compression can degrade high-frequency synthetic cues.

### Q14: What would you improve in future work?
**Answer**:  
1. Multi-generator training incorporating samples from GANs, Latent Diffusion, and Flux.
2. Dual-stream networks combining spatial CNN features with Frequency-Domain analysis (Discrete Cosine Transform / FFT).
3. Visual explainability by integrating Grad-CAM heatmaps to visually highlight which regions of an image triggered the synthetic classification.
