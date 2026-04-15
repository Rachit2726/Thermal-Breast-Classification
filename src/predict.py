"""
Predict on a single thermal image using the trained ensemble.
Usage:
    python predict.py --image path/to/image.jpg
"""
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from config_loader import load_config


def predict(image_path):
    cfg = load_config()
    img_size = tuple(cfg["image"]["size"])
    model_paths = [cfg["models"][m]["save_path"] for m in cfg["models"]]
    threshold = cfg["training"]["threshold"]

    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    probs = []
    for path in model_paths:
        model = tf.keras.models.load_model(path)
        probs.append(model.predict(img_array, verbose=0).ravel()[0])

    avg_prob = np.mean(probs)
    label = "Malignant" if avg_prob > threshold else "Benign"

    print(f"Image       : {image_path}")
    print(f"Probability : {avg_prob:.4f}")
    print(f"Prediction  : {label}")
    print(f"Threshold   : {threshold}")
    return label, avg_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict breast cancer from thermal image")
    parser.add_argument("--image", required=True, help="Path to thermal image")
    args = parser.parse_args()
    predict(args.image)
