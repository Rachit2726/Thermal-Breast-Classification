"""
Evaluate ensemble of all 3 models on the test set.
Generates confusion matrix, ROC curve, and prints all metrics.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, roc_curve, ConfusionMatrixDisplay
)
from config_loader import load_config


def evaluate():
    cfg = load_config()
    t_cfg = cfg["training"]
    img_size = tuple(cfg["image"]["size"])
    test_dir = os.path.join(cfg["data"]["final_dir"], "test")
    out_dir = cfg["results"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    model_paths = [cfg["models"][m]["save_path"] for m in cfg["models"]]

    test_gen = ImageDataGenerator(rescale=1./255)
    test_data = test_gen.flow_from_directory(
        test_dir, target_size=img_size,
        batch_size=t_cfg["batch_size"], class_mode="binary", shuffle=False
    )
    y_true = test_data.classes

    # Ensemble predictions
    all_probs = []
    for path in model_paths:
        print(f"Loading: {path}")
        model = tf.keras.models.load_model(path)
        all_probs.append(model.predict(test_data).ravel())

    y_probs = np.mean(all_probs, axis=0)
    y_pred = (y_probs > t_cfg["threshold"]).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Confusion Matrix – Ensemble")
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Ensemble (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    dice = (2 * tp) / (2 * tp + fp + fn)
    mcc = matthews_corrcoef(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])

    metrics_text = (
        f"Classification Report:\n{report}\n"
        f"Accuracy              : {accuracy:.4f}\n"
        f"ROC-AUC               : {auc:.4f}\n"
        f"Specificity (TNR)     : {specificity:.4f}\n"
        f"IoU (Jaccard Index)   : {iou:.4f}\n"
        f"Dice Coefficient      : {dice:.4f}\n"
        f"Matthews CC (MCC)     : {mcc:.4f}\n"
    )

    print(metrics_text)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(metrics_text)

    print(f"✅ Results saved to {out_dir}/")


if __name__ == "__main__":
    evaluate()
