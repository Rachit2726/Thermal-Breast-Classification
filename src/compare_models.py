"""
Compare individual model performances on the test set.
Generates a grouped bar chart saved to results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from config_loader import load_config


def compare():
    cfg = load_config()
    t_cfg = cfg["training"]
    img_size = tuple(cfg["image"]["size"])
    test_dir = os.path.join(cfg["data"]["final_dir"], "test")
    out_dir = cfg["results"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    test_gen = ImageDataGenerator(rescale=1./255)
    test_data = test_gen.flow_from_directory(
        test_dir, target_size=img_size,
        batch_size=t_cfg["batch_size"], class_mode="binary", shuffle=False
    )
    y_true = test_data.classes

    results = {}
    for key in cfg["models"]:
        path = cfg["models"][key]["save_path"]
        name = cfg["models"][key]["name"]
        print(f"Evaluating {name}...")
        model = tf.keras.models.load_model(path)
        y_probs = model.predict(test_data).ravel()
        y_pred = (y_probs > t_cfg["threshold"]).astype(int)

        results[name] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_probs),
        }

    # Plot grouped bar chart
    metrics = list(next(iter(results.values())).keys())
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=300)
    plt.close(fig)

    # Print table
    print(f"\n{'Model':<18}", end="")
    for m in metrics:
        print(f"{m:<12}", end="")
    print()
    print("-" * 78)
    for name in model_names:
        print(f"{name:<18}", end="")
        for m in metrics:
            print(f"{results[name][m]:<12.4f}", end="")
        print()

    print(f"\n✅ Comparison chart saved to {out_dir}/model_comparison.png")


if __name__ == "__main__":
    compare()
