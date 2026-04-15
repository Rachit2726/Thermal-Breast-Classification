"""
Preprocess raw thermal images: convert to RGB, resize, normalize, and save.
"""
import os
from PIL import Image
import numpy as np
from config_loader import load_config


def preprocess_images(cfg=None):
    if cfg is None:
        cfg = load_config()

    input_dir = cfg["data"]["raw_dir"]
    output_dir = cfg["data"]["preprocessed_dir"]
    img_size = tuple(cfg["image"]["preprocess_size"])

    os.makedirs(output_dir, exist_ok=True)

    for label in cfg["data"]["classes"]:
        in_path = os.path.join(input_dir, label)
        out_path = os.path.join(output_dir, label)
        os.makedirs(out_path, exist_ok=True)

        for img_name in os.listdir(in_path):
            img = Image.open(os.path.join(in_path, img_name)).convert("RGB")
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            img_out = Image.fromarray((img_array * 255).astype(np.uint8))
            img_out.save(os.path.join(out_path, img_name))

    print("✅ Preprocessing complete")


if __name__ == "__main__":
    preprocess_images()
