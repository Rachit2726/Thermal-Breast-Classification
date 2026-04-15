"""
Split real + synthetic data into train/val/test sets.
"""
import os
import shutil
import random
from config_loader import load_config


def split_dataset(cfg=None):
    if cfg is None:
        cfg = load_config()

    real_dir = cfg["data"]["preprocessed_dir"]
    synth_dir = cfg["data"]["synthetic_dir"]
    final_dir = cfg["data"]["final_dir"]
    classes = cfg["data"]["classes"]
    split = cfg["data"]["split"]

    for s in split:
        for cls in classes:
            os.makedirs(os.path.join(final_dir, s, cls), exist_ok=True)

    # Split real data
    for cls in classes:
        imgs = [f for f in os.listdir(os.path.join(real_dir, cls))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(imgs)
        total = len(imgs)
        train_end = int(split["train"] * total)
        val_end = train_end + int(split["val"] * total)

        split_imgs = {
            "train": imgs[:train_end],
            "val": imgs[train_end:val_end],
            "test": imgs[val_end:]
        }

        for s, files in split_imgs.items():
            for f in files:
                shutil.copy(os.path.join(real_dir, cls, f), os.path.join(final_dir, s, cls, f))

    print("✅ Real data split completed")

    # Add synthetic data to train only
    for cls in classes:
        synth_imgs = [f for f in os.listdir(os.path.join(synth_dir, cls))
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for f in synth_imgs:
            shutil.copy(os.path.join(synth_dir, cls, f), os.path.join(final_dir, "train", cls, f))

    print("✅ Synthetic data added to TRAIN only")

    # Summary
    print("\n📊 FINAL DATASET SUMMARY")
    for s in ["train", "val", "test"]:
        for cls in classes:
            count = len(os.listdir(os.path.join(final_dir, s, cls)))
            print(f"{s}/{cls}: {count} images")


if __name__ == "__main__":
    split_dataset()
