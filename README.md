# 🩺 Thermal Breast Cancer Classification using Ensemble CNNs with Synthetic Data Augmentation

> **Final Year B.Tech Project**
>
> A deep learning pipeline that classifies breast thermal (infrared) images as **Benign** or **Malignant** using an ensemble of fine-tuned CNNs (ResNet50, EfficientNetB0, MobileNetV2), augmented with Stable Diffusion-generated synthetic data.

---

## 📁 Project Structure

```
Thermal breast classification/
│
├── config/
│   └── config.yaml              # Centralized project configuration
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py          # YAML config loader utility
│   ├── preprocess.py             # Step 1: Preprocess raw thermal images
│   ├── generate_img.py           # Step 2: Generate synthetic images (Stable Diffusion)
│   ├── split_and_augment.py      # Step 3: Split data into train/val/test
│   ├── train.py                  # Step 4: Train models (ResNet50 / EfficientNet / MobileNet)
│   ├── evaluate.py               # Step 5: Ensemble evaluation with metrics & plots
│   ├── predict.py                # Single image prediction
│   └── compare_models.py         # Compare individual model performances
│
├── app/
│   └── app.py                    # Streamlit web app for live demo
│
├── raw_data/                     # Original thermal images
│   ├── benign/
│   └── malignant/
│
├── synthetic_data/               # Preprocessed images (256×256)
│   ├── benign/
│   └── malignant/
│
├── synthetic_data1/              # Stable Diffusion generated images
│   ├── benign/
│   └── malignant/
│
├── final_dataset/                # Train/Val/Test split (ready for training)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                       # Saved trained model weights
├── results/                      # Evaluation outputs (plots, metrics)
├── logs/                         # Training CSV logs
├── notebooks/                    # Jupyter notebooks (experiments)
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🧠 Models Used

| Model | Type | Fine-Tuned Layers |
|---|---|---|
| **ResNet50** | Transfer Learning | Last 30 layers |
| **EfficientNetB0** | Transfer Learning | Last 30 layers |
| **MobileNetV2** | Transfer Learning | Last 30 layers |

Final prediction is made by **averaging probabilities** from all 3 models (ensemble).

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Specificity (TNR)
- IoU (Jaccard Index)
- Dice Coefficient
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix & ROC Curve plots

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9 or 3.10 (recommended)
- pip
- GPU recommended for training (CPU works but is slow)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "Thermal breast classification"
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run (Step by Step)

> **All commands should be run from the project root directory.**
> **All `src/` scripts should be run from inside the `src/` folder:**

```bash
cd src
```

### Step 1 — Preprocess Raw Images

Converts raw thermal images to RGB, resizes to 256×256, and normalizes.

```bash
python preprocess.py
```

**Input:** `raw_data/benign/`, `raw_data/malignant/`
**Output:** `synthetic_data/benign/`, `synthetic_data/malignant/`

---

### Step 2 — Generate Synthetic Images (Optional)

Uses Stable Diffusion img2img to generate synthetic thermal images for data augmentation.

```bash
python generate_img.py
```

> ⚠️ This step requires ~4GB RAM and takes significant time on CPU. Skip if you already have `synthetic_data1/` populated.

**Input:** `synthetic_data/`
**Output:** `synthetic_data1/`

---

### Step 3 — Split Dataset

Splits real + synthetic data into 70/15/15 train/val/test. Synthetic images go to train only.

```bash
python split_and_augment.py
```

**Output:** `final_dataset/train/`, `final_dataset/val/`, `final_dataset/test/`

---

### Step 4 — Train Models

Train each model individually:

```bash
python train.py --model resnet
python train.py --model efficientnet
python train.py --model mobilenet
```

**Output:** Trained models saved to `models/` folder, training logs to `logs/`.

---

### Step 5 — Evaluate Ensemble

Runs ensemble evaluation on the test set and generates plots + metrics.

```bash
python evaluate.py
```

**Output (in `results/`):**
- `confusion_matrix.png`
- `roc_curve.png`
- `metrics.txt`

---

### Step 6 — Compare Models (Optional)

Generates a grouped bar chart comparing individual model performances.

```bash
python compare_models.py
```

**Output:** `results/model_comparison.png`

---

### Single Image Prediction

```bash
python predict.py --image path/to/thermal_image.jpg
```

---

## 🌐 Web App (Streamlit)

A live demo app for presentations. Run from the **project root**:

```bash
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser.

**Features:**
- Upload any thermal breast image
- See individual model predictions
- See ensemble result with probability bar
- Clean UI for project demo/viva

---

## 📂 Dataset Info

- **Source:** Infrared thermal breast images (IIR dataset)
- **Classes:** Benign, Malignant
- **Augmentation:** Stable Diffusion v1.5 (img2img) synthetic generation
- **Split:** 70% Train / 15% Val / 15% Test
- **Synthetic data** is added to the training set only (no data leakage)

---

## 🔧 Configuration

All hyperparameters and paths are centralized in `config/config.yaml`. Modify this file to change:

- Image sizes
- Batch size, epochs, learning rate
- Train/val/test split ratios
- Model save paths
- Stable Diffusion generation parameters

---

## 📝 Key Highlights (For Viva)

1. **Ensemble Learning** — Combines 3 architectures for robust predictions
2. **Synthetic Data Augmentation** — Uses generative AI (Stable Diffusion) to handle class imbalance
3. **Transfer Learning** — Fine-tunes pretrained ImageNet models on medical data
4. **Class Weighting** — Handles imbalanced benign/malignant distribution
5. **Threshold Tuning** — Uses 0.35 threshold (not 0.5) to prioritize recall for malignant cases
6. **Web Application** — Streamlit app for real-time inference demo
7. **Comprehensive Metrics** — Goes beyond accuracy with AUC, MCC, Dice, IoU, Specificity

---

## 📜 License

This project is for academic/educational purposes only.

---

## 👥 Authors

- Final Year B.Tech Students
- Department of Computer Science & Engineering
