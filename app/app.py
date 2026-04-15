"""
Thermal Breast Cancer Classification — Streamlit Demo App
Run from project root: streamlit run app/app.py
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import streamlit as st
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from PIL import Image
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup so src/ imports work ──────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from gradcam import generate_gradcam

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThermoScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- design tokens ---- */
:root {
    --bg-main:      #0f1117;
    --bg-card:      #1a1f2e;
    --bg-secondary: #161b27;
    --primary:      #4a9eda;
    --success:      #52c07a;
    --danger:       #e05252;
    --text-main:    #e8f4fd;
    --text-secondary: #8ab4d4;
    --border:       #2a3040;
}

/* ---- global ---- */
[data-testid="stAppViewContainer"] { background: var(--bg-main); }
[data-testid="stSidebar"] { background: var(--bg-secondary); border-right: 1px solid var(--border); }

/* ---- typography ---- */
h1, h2, h3 { font-weight: 600; letter-spacing: 0.3px; }
.subtitle { color: var(--text-secondary); font-size: 0.9rem; margin: 0; }

/* ---- header banner ---- */
.banner {
    background: linear-gradient(135deg, #1a1f35 0%, #0d2137 100%);
    border: 1px solid #2a4a6b;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.banner h1 { color: var(--text-main); font-size: 2rem; margin: 0 0 6px 0; }
.banner p  { color: var(--text-secondary); margin: 0; font-size: 0.95rem; }

/* ---- section spacing ---- */
.section { margin-top: 30px; margin-bottom: 30px; }

/* ---- result cards ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 16px;
    line-height: 1.5;
}
.card-malignant { border-left: 4px solid var(--danger); }
.card-benign    { border-left: 4px solid var(--success); }

/* ---- verdict box ---- */
.verdict-malignant {
    background: linear-gradient(135deg, #2d1515, #1a0d0d);
    border: 2px solid var(--danger);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,0,0,0.3);
}
.verdict-benign {
    background: linear-gradient(135deg, #0d2d1a, #0a1f12);
    border: 2px solid var(--success);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,0,0,0.3);
}
.verdict-label { font-size: 2rem; font-weight: 700; margin: 0; }
.verdict-sub   { color: #aaa; font-size: 0.85rem; margin-top: 6px; }

/* ---- metric pill ---- */
.pill {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2e3a50;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 3px;
}

/* ---- info badge ---- */
.badge {
    background: #1a2a3a;
    border: 1px solid #2a4a6b;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #c8dff0;
    font-size: 0.88rem;
}

/* ---- tab styling ---- */
[data-testid="stTabs"] button { color: var(--text-secondary) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text-main) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* ---- progress bar ---- */
.stProgress > div > div { background: var(--primary); }

/* ---- hide streamlit branding ---- */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── config & model loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading configuration…")
def load_cfg():
    cfg_path = os.path.join(ROOT, "config", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading models…")
def load_models(cfg):
    models = {}
    for key, val in cfg["models"].items():
        path = os.path.join(ROOT, val["save_path"])
        if os.path.exists(path):
            models[val["name"]] = tf.keras.models.load_model(path, compile=False)
    return models


cfg = load_cfg()
models = load_models(cfg)
IMG_SIZE = tuple(cfg["image"]["size"])
THRESHOLD = cfg["training"]["threshold"]

# ── dataset stats (computed once) ────────────────────────────────────────────
@st.cache_data
def dataset_counts():
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for cls in ("benign", "malignant"):
            d = os.path.join(ROOT, cfg["data"]["final_dir"], split, cls)
            counts[split][cls] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return counts


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <h2 style="color:#e8f4fd;">ThermoScan AI</h2>
    <p class="subtitle">Clinical AI Assistant</p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Models loaded**")
    for name in models:
        st.markdown(f"✅ {name}")
    if not models:
        st.error("No models found in `models/`")

    st.markdown("---")
    st.markdown("**Decision threshold**")
    threshold_display = st.slider(
        "Malignant threshold", 0.1, 0.9, THRESHOLD, 0.01,
        help="Probability above this → Malignant"
    )
    st.markdown("---")
    st.markdown(
        "<span class='pill'>ResNet50</span>"
        "<span class='pill'>EfficientNetB0</span>"
        "<span class='pill'>MobileNetV2</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Final Year B.Tech Project · CSE")


# ── banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>Thermal Breast Cancer Classification</h1>
  <p class="subtitle">AI-powered diagnostic support using ensemble deep learning</p>
</div>
""", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_about = st.tabs(
    ["🔬 Predict", "📊 Model Insights", "ℹ️ About"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload Thermal Image")
        uploaded = st.file_uploader(
            "Drag & drop or browse", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        show_gradcam = st.checkbox("Show GradCAM heatmap", value=True)

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)

    with col_result:
        if not uploaded:
            st.markdown("""
            <div style="height:320px;display:flex;align-items:center;
                        justify-content:center;border:2px dashed #2a3a50;
                        border-radius:12px;color:#4a6a8a;font-size:1rem;">
                ← Upload an image to see predictions
            </div>
            """, unsafe_allow_html=True)
        else:
            img_resized = img.resize(IMG_SIZE)
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            st.markdown("""
            <div class="section">
            <h3 style="color:#e8f4fd;">Model Predictions</h3>
            <p class="subtitle">Individual model outputs and confidence scores</p>
            </div>
            """, unsafe_allow_html=True)
            probs = []
            with st.spinner("Running inference…"):
                for name, model in models.items():
                    p = float(model.predict(img_array, verbose=0).ravel()[0])
                    probs.append(p)
                    label = "Malignant" if p > threshold_display else "Benign"
                    color = "#e05252" if label == "Malignant" else "#52c07a"
                    bar_pct = int(p * 100)
                    st.markdown(f"""
                    <div class="card card-{'malignant' if label=='Malignant' else 'benign'}">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#c8dff0;font-weight:600">{name}</span>
                        <span style="color:{color};font-weight:700">{label}</span>
                      </div>
                      <div style="margin-top:8px;background:#0f1117;border-radius:6px;height:8px">
                        <div style="width:{bar_pct}%;background:{color};height:8px;
                                    border-radius:6px;transition:width 0.4s"></div>
                      </div>
                      <div style="color:#6a8aaa;font-size:0.8rem;margin-top:4px">
                        Probability: {p:.4f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Ensemble verdict
            avg = float(np.mean(probs))
            is_malignant = avg > threshold_display
            verdict_class = "verdict-malignant" if is_malignant else "verdict-benign"
            verdict_icon = "🔴" if is_malignant else "🟢"
            verdict_text = "MALIGNANT" if is_malignant else "BENIGN"
            verdict_color = "#e05252" if is_malignant else "#52c07a"

            st.markdown("<hr style='border:1px solid #2a3040;'>", unsafe_allow_html=True)
            st.markdown("""
            <h3 style="color:#e8f4fd;">Ensemble Verdict</h3>
            <p class="subtitle">Combined prediction from all 3 models</p>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="{verdict_class}">
              <p class="verdict-label" style="color:{verdict_color}">
                {verdict_icon} {verdict_text}
              </p>
              <p class="verdict-sub">
                Ensemble probability: <strong style="color:{verdict_color}">{avg:.4f}</strong>
                &nbsp;·&nbsp; Threshold: {threshold_display:.2f}
              </p>
            </div>
            """, unsafe_allow_html=True)

            if is_malignant:
                st.warning(
                    "⚠️ This is a screening tool only. "
                    "Please consult a qualified medical professional for diagnosis.",
                    icon="⚠️"
                )

    # GradCAM section (full width below)
    if uploaded and show_gradcam and models:
        st.markdown("<hr style='border:1px solid #2a3040;'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="color:#e8f4fd;">GradCAM Explainability</h3>
        <p class="subtitle">Gradient-weighted Class Activation Maps highlight regions the model focuses on</p>
        """, unsafe_allow_html=True)

        gcam_cols = st.columns(len(models))
        img_array_gcam = np.expand_dims(np.array(img.resize(IMG_SIZE)) / 255.0, axis=0)
        for col, (name, model) in zip(gcam_cols, models.items()):
            with col:
                try:
                    overlay, heatmap = generate_gradcam(model, img_array_gcam)
                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                    fig.patch.set_facecolor("#0f1117")
                    for ax in axes:
                        ax.set_facecolor("#0f1117")
                        ax.axis("off")
                    axes[0].imshow(overlay)
                    axes[0].set_title("Overlay", color="#8ab4d4", fontsize=9)
                    axes[1].imshow(heatmap)
                    axes[1].set_title("Heatmap", color="#8ab4d4", fontsize=9)
                    fig.suptitle(name, color="#c8dff0", fontsize=10, y=1.02)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"{name}: GradCAM failed — {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("""
    <div class="section">
    <h3 style="color:#e8f4fd;">Dataset Distribution</h3>
    <p class="subtitle">Image counts across train / val / test splits</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        counts = dataset_counts()
        split_labels = list(counts.keys())
        benign_vals = [counts[s]["benign"] for s in split_labels]
        malignant_vals = [counts[s]["malignant"] for s in split_labels]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b27")
        x = np.arange(len(split_labels))
        w = 0.35
        b1 = ax.bar(x - w/2, benign_vals, w, label="Benign", color="#52c07a", alpha=0.85)
        b2 = ax.bar(x + w/2, malignant_vals, w, label="Malignant", color="#e05252", alpha=0.85)
        ax.bar_label(b1, padding=3, color="#c8dff0", fontsize=9)
        ax.bar_label(b2, padding=3, color="#c8dff0", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in split_labels], color="#8ab4d4")
        ax.set_ylabel("Images", color="#8ab4d4")
        ax.tick_params(colors="#8ab4d4")
        ax.spines[:].set_color("#2a3040")
        ax.legend(facecolor="#1a1f2e", labelcolor="#c8dff0", edgecolor="#2a3040")
        ax.set_title("Train / Val / Test Split", color="#c8dff0", pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Summary pills
        total = sum(benign_vals) + sum(malignant_vals)
        st.markdown(
            f"<span class='pill'>Total images: {total}</span>"
            f"<span class='pill'>Benign: {sum(benign_vals)}</span>"
            f"<span class='pill'>Malignant: {sum(malignant_vals)}</span>"
            f"<span class='pill'>Split: 70 / 15 / 15</span>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Could not load dataset stats: {e}")

    st.markdown("<hr style='border:1px solid #2a3040;'>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style="color:#e8f4fd;">Model Performance</h3>
    <p class="subtitle">Confusion matrix, ROC curve, and all evaluation metrics</p>
    """, unsafe_allow_html=True)

    results_dir = os.path.join(ROOT, cfg["results"]["output_dir"])
    cm_path  = os.path.join(results_dir, "confusion_matrix.png")
    roc_path = os.path.join(results_dir, "roc_curve.png")
    cmp_path = os.path.join(results_dir, "model_comparison.png")
    met_path = os.path.join(results_dir, "metrics.txt")

    has_results = any(os.path.exists(p) for p in [cm_path, roc_path, cmp_path, met_path])

    if not has_results:
        st.info(
            "No evaluation results found yet. "
            "Run `python src/evaluate.py` and `python src/compare_models.py` "
            "from the project root to generate them.",
            icon="ℹ️"
        )
    else:
        # ── Metric cards parsed from metrics.txt ─────────────────────────────
        if os.path.exists(met_path):
            with open(met_path) as f:
                metrics_text = f.read()

            import re
            def _extract(pattern, text, default="N/A"):
                m = re.search(pattern, text)
                return f"{float(m.group(1)):.4f}" if m else default

            # parse per-class precision/recall/f1 from classification report
            def _cls_metric(cls, metric, text):
                patterns = {
                    "precision": rf"{cls}\s+([0-9.]+)",
                    "recall":    rf"{cls}\s+[0-9.]+\s+([0-9.]+)",
                    "f1":        rf"{cls}\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
                }
                m = re.search(patterns[metric], text)
                return f"{float(m.group(1)):.4f}" if m else "N/A"

            scores = {
                "Accuracy":    _extract(r"Accuracy\s*:\s*([0-9.]+)", metrics_text),
                "ROC-AUC":     _extract(r"ROC-AUC\s*:\s*([0-9.]+)", metrics_text),
                "Specificity": _extract(r"Specificity[^:]*:\s*([0-9.]+)", metrics_text),
                "IoU":         _extract(r"IoU[^:]*:\s*([0-9.]+)", metrics_text),
                "Dice":        _extract(r"Dice[^:]*:\s*([0-9.]+)", metrics_text),
                "MCC":         _extract(r"Matthews[^:]*:\s*([0-9.]+)", metrics_text),
                "Precision (B)": _cls_metric("Benign",    "precision", metrics_text),
                "Recall (B)":    _cls_metric("Benign",    "recall",    metrics_text),
                "F1 (B)":        _cls_metric("Benign",    "f1",        metrics_text),
                "Precision (M)": _cls_metric("Malignant", "precision", metrics_text),
                "Recall (M)":    _cls_metric("Malignant", "recall",    metrics_text),
                "F1 (M)":        _cls_metric("Malignant", "f1",        metrics_text),
            }

            icons = {
                "Accuracy": "🎯", "ROC-AUC": "📈", "Specificity": "🛡️",
                "IoU": "🔲", "Dice": "🎲", "MCC": "⚖️",
                "Precision (B)": "🟢", "Recall (B)": "🟢", "F1 (B)": "🟢",
                "Precision (M)": "🔴", "Recall (M)": "🔴", "F1 (M)": "🔴",
            }

            st.markdown("##### 📊 Ensemble Scores")
            cols = st.columns(6)
            for i, (label, val) in enumerate(scores.items()):
                with cols[i % 6]:
                    st.markdown(f"""
                    <div style="background:#1a1f2e;border:1px solid #2a3040;
                                border-radius:10px;padding:14px 10px;
                                text-align:center;margin-bottom:10px">
                      <div style="font-size:1.3rem">{icons[label]}</div>
                      <div style="color:#4a9eda;font-size:1.1rem;
                                  font-weight:700;margin:4px 0">{val}</div>
                      <div style="color:#6a8aaa;font-size:0.72rem">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📋 Full Metrics Report"):
                st.code(metrics_text, language="text")

        # ── Confusion Matrix + ROC ────────────────────────────────────────────
        st.markdown("##### 🗺️ Confusion Matrix & ROC Curve")
        r1, r2 = st.columns(2)
        if os.path.exists(cm_path):
            r1.image(cm_path, caption="Confusion Matrix — Ensemble", use_container_width=True)
        if os.path.exists(roc_path):
            r2.image(roc_path, caption="ROC Curve — Ensemble", use_container_width=True)

        if os.path.exists(cmp_path):
            st.markdown("##### 🏆 Individual Model Comparison")
            st.image(cmp_path, caption="Model Comparison", use_container_width=True)

    st.markdown("<hr style='border:1px solid #2a3040;'>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style="color:#e8f4fd;">Architecture Summary</h3>
    <p class="subtitle">Model parameters and fine-tuning configuration</p>
    """, unsafe_allow_html=True)
    arch_data = {
        "Model": ["ResNet50", "EfficientNetB0", "MobileNetV2"],
        "Params (M)": ["~25.6", "~5.3", "~3.4"],
        "Fine-tuned Layers": ["Last 30", "Last 30", "Last 30"],
        "Pretrained On": ["ImageNet", "ImageNet", "ImageNet"],
        "Input Size": ["224×224", "224×224", "224×224"],
    }
    st.table(arch_data)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("#### Project Overview")
        st.markdown("""
        This system classifies **infrared thermal breast images** as Benign or Malignant
        using an ensemble of three fine-tuned convolutional neural networks.

        **Key innovations:**
        """)
        highlights = [
            ("🧠 Ensemble Learning",
             "Averages probabilities from ResNet50, EfficientNetB0, and MobileNetV2 "
             "for robust, low-variance predictions."),
            ("🎨 Synthetic Augmentation",
             "Stable Diffusion v1.5 img2img generates additional training samples, "
             "addressing class imbalance without data leakage."),
            ("🔥 GradCAM Explainability",
             "Gradient-weighted Class Activation Maps visualise which thermal regions "
             "drive each model's decision — critical for clinical trust."),
            ("⚖️ Threshold Tuning",
             "Decision threshold set to 0.35 (not 0.5) to maximise recall for "
             "malignant cases, prioritising sensitivity in a medical context."),
            ("📊 Comprehensive Metrics",
             "Accuracy, AUC, MCC, Dice, IoU, Specificity — going well beyond "
             "simple accuracy for a rigorous evaluation."),
        ]
        for title, desc in highlights:
            st.markdown(f"""
            <div class="badge">
              <strong>{title}</strong><br>{desc}
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Pipeline")
        steps = [
            ("1", "Preprocess", "Raw → 256×256 RGB normalised"),
            ("2", "Synthesise", "Stable Diffusion img2img augmentation"),
            ("3", "Split", "70 / 15 / 15 train / val / test"),
            ("4", "Train", "Fine-tune 3 CNNs with class weights"),
            ("5", "Ensemble", "Average probabilities, threshold 0.35"),
            ("6", "Explain", "GradCAM heatmaps per model"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;margin-bottom:12px">
              <div style="background:#1a3a5a;color:#4a9eda;font-weight:700;
                          border-radius:50%;width:28px;height:28px;min-width:28px;
                          display:flex;align-items:center;justify-content:center;
                          margin-right:12px;font-size:0.85rem">{num}</div>
              <div>
                <div style="color:#c8dff0;font-weight:600;font-size:0.9rem">{title}</div>
                <div style="color:#6a8aaa;font-size:0.8rem">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Tech Stack")
        stack = [
            "TensorFlow / Keras", "Stable Diffusion (diffusers)",
            "Streamlit", "scikit-learn", "OpenCV", "NumPy / Matplotlib",
        ]
        st.markdown(
            "".join(f"<span class='pill'>{s}</span>" for s in stack),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Dataset")
        st.markdown("""
        <div class="badge">
          <strong>IIR Thermal Breast Dataset</strong><br>
          119 patients · Benign & Malignant classes<br>
          Anterior + oblique views per patient<br>
          Augmented with 357 synthetic images via Stable Diffusion
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "⚠️ For academic/research purposes only. "
        "Not a substitute for professional medical diagnosis. "
        "· Final Year B.Tech Project · Department of Computer Science & Engineering"
    )
