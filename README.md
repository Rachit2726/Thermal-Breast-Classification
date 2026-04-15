# 🩺 Thermal Breast Cancer Classification

An ensemble deep learning pipeline that classifies breast thermal (infrared) images as **Benign** or **Malignant** using fine-tuned ResNet50, EfficientNetB0, and MobileNetV2, augmented with Stable Diffusion-generated synthetic data.

---

## 📋 Prerequisites

- Python **3.10** (recommended — TensorFlow 2.13 does not support Python 3.11+)
- `pip`
- Git
- ~5 GB free disk space (models + synthetic data)
- GPU optional (CPU works, but training and Stable Diffusion generation will be slow)

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Thermal breast classification"
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

> You should see `(venv)` at the start of your terminal prompt once activated.
> To deactivate at any time, run `deactivate`.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ If you have an NVIDIA GPU, replace the three `+cpu` lines in `requirements.txt` with the matching CUDA versions from https://pytorch.org/get-started/locally/ before installing.

---

## 🚀 Running the Pipeline

> All `src/` scripts must be run from **inside the `src/` folder**.

```bash
cd src
```

---

### Step 1 — Preprocess raw images

Converts raw thermal images to 256×256 RGB and normalises them.

```bash
python preprocess.py
```

| | Path |
|---|---|
| Input | `raw_data/benign/`, `raw_data/malignant/` |
| Output | `synthetic_data/benign/`, `synthetic_data/malignant/` |

---

### Step 2 — Generate synthetic images *(optional)*

Uses Stable Diffusion img2img to create additional training samples.

```bash
python generate_img.py
```

| | Path |
|---|---|
| Input | `synthetic_data/` |
| Output | `synthetic_data1/` |

> ⚠️ This step downloads the `runwayml/stable-diffusion-v1-5` model (~4 GB) on first run and is slow on CPU. Skip if `synthetic_data1/` is already populated.

---

### Step 3 — Split dataset

Splits real + synthetic images into 70 / 15 / 15 train / val / test. Synthetic images go to train only (no data leakage).

```bash
python split_and_augment.py
```

| | Path |
|---|---|
| Output | `final_dataset/train/`, `final_dataset/val/`, `final_dataset/test/` |

---

### Step 4 — Train models

Train each architecture individually:

```bash
python train.py --model resnet
python train.py --model efficientnet
python train.py --model mobilenet
```

| | Path |
|---|---|
| Output models | `models/resnet50.keras`, `models/efficientnet.keras`, `models/mobilenet.keras` |
| Training logs | `logs/<model>_training.csv` |

---

### Step 5 — Evaluate ensemble

Runs ensemble inference on the test set and saves all metrics and plots.

```bash
python evaluate.py
```

| Output file | Description |
|---|---|
| `results/confusion_matrix.png` | Confusion matrix |
| `results/roc_curve.png` | ROC curve |
| `results/metrics.txt` | Accuracy, AUC, MCC, Dice, IoU, Specificity |

---

### Step 6 — Compare individual models *(optional)*

Generates a grouped bar chart comparing the three models side by side.

```bash
python compare_models.py
```

| | Path |
|---|---|
| Output | `results/model_comparison.png` |

---

### Single image prediction

```bash
python predict.py --image path/to/thermal_image.jpg
```

---

## 🌐 Streamlit Web App

Run from the **project root** (not `src/`):

```bash
cd ..          # if you are still inside src/
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

**Features:**
- Upload any thermal breast image
- Per-model prediction bars with confidence scores
- Ensemble verdict with probability
- GradCAM heatmaps for explainability
- Dataset distribution chart and evaluation metrics dashboard

---

## 🔧 Configuration

All hyperparameters and paths live in `config/config.yaml`. Key settings:

| Key | Default | Description |
|---|---|---|
| `training.epochs` | 25 | Max training epochs |
| `training.batch_size` | 16 | Batch size |
| `training.learning_rate` | 0.00001 | Adam LR |
| `training.threshold` | 0.35 | Malignant decision threshold |
| `training.fine_tune_layers` | 30 | Unfrozen layers from top |
| `image.size` | [224, 224] | Model input size |
| `generation.device` | cpu | `cpu` or `cuda` for Stable Diffusion |

---

## 📁 Project Structure

```
Thermal breast classification/
├── config/config.yaml          # Centralised configuration
├── src/
│   ├── preprocess.py           # Step 1
│   ├── generate_img.py         # Step 2
│   ├── split_and_augment.py    # Step 3
│   ├── train.py                # Step 4
│   ├── evaluate.py             # Step 5
│   ├── compare_models.py       # Step 6
│   ├── predict.py              # Single image inference
│   ├── gradcam.py              # GradCAM explainability
│   └── config_loader.py        # YAML loader
├── app/app.py                  # Streamlit demo
├── raw_data/                   # Original thermal images
├── synthetic_data/             # Preprocessed 256×256 images
├── synthetic_data1/            # Stable Diffusion outputs
├── final_dataset/              # Train / Val / Test split
├── models/                     # Saved .keras model weights
├── results/                    # Plots and metrics
├── logs/                       # CSV training logs
├── requirements.txt
└── README.md
```

---

## 🧠 Models

| Model | Parameters | Fine-tuned Layers | Pretrained On |
|---|---|---|---|
| ResNet50 | ~25.6 M | Last 30 | ImageNet |
| EfficientNetB0 | ~5.3 M | Last 30 | ImageNet |
| MobileNetV2 | ~3.4 M | Last 30 | ImageNet |

Final prediction = average of all three model probabilities (soft voting ensemble).

---

## ⚠️ Disclaimer

For academic / research purposes only. Not a substitute for professional medical diagnosis.

**Final Year B.Tech Project · Department of Computer Science & Engineering**
