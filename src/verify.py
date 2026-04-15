"""
End-to-end pipeline verification script.
Checks dataset integrity, model files, config, and runs a smoke test.

Usage:
    python verify.py
    python verify.py --full   (also runs a model smoke test)
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

# ── allow running from src/ or project root ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

errors = []


def check(label, condition, fatal=True, warn_msg=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        tag = FAIL if fatal else WARN
        msg = warn_msg or label
        print(f"{tag} {msg}")
        if fatal:
            errors.append(label)


def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def verify_config(cfg):
    section("1. Config")
    check("config.yaml loaded", cfg is not None)
    check("data.classes defined", len(cfg["data"]["classes"]) == 2)
    check("training.threshold in (0,1)", 0 < cfg["training"]["threshold"] < 1)
    check("image.size is [224,224]", cfg["image"]["size"] == [224, 224])


def verify_dataset(cfg):
    section("2. Dataset")
    final_dir = cfg["data"]["final_dir"]
    classes = cfg["data"]["classes"]

    for split in ["train", "val", "test"]:
        for cls in classes:
            path = os.path.join(final_dir, split, cls)
            exists = os.path.isdir(path)
            count = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if exists else 0
            check(f"{split}/{cls}: {count} images", exists and count > 0)

    # Check for corrupt images
    section("2b. Image Integrity (sample check)")
    corrupt = 0
    total_checked = 0
    for split in ["train", "val", "test"]:
        for cls in classes:
            path = os.path.join(final_dir, split, cls)
            if not os.path.isdir(path):
                continue
            files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            sample = files[:10]
            for f in sample:
                total_checked += 1
                try:
                    img = Image.open(os.path.join(path, f))
                    img.verify()
                except Exception:
                    corrupt += 1

    check(f"Image integrity ({total_checked} sampled, {corrupt} corrupt)", corrupt == 0,
          fatal=False, warn_msg=f"{corrupt} corrupt images found in sample")


def verify_models(cfg):
    section("3. Model Files")
    for key in cfg["models"]:
        path = cfg["models"][key]["save_path"]
        name = cfg["models"][key]["name"]
        exists = os.path.isfile(path)
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        check(f"{name} — {path} ({size_mb:.1f} MB)", exists and size_mb > 1)


def verify_results(cfg):
    section("4. Results")
    out_dir = cfg["results"]["output_dir"]
    for fname in ["confusion_matrix.png", "roc_curve.png", "metrics.txt"]:
        path = os.path.join(out_dir, fname)
        check(f"results/{fname} exists", os.path.isfile(path),
              fatal=False, warn_msg=f"results/{fname} missing — run evaluate.py first")


def smoke_test(cfg):
    section("5. Model Smoke Test (inference on dummy image)")
    try:
        import tensorflow as tf
        img_size = tuple(cfg["image"]["size"])
        dummy = np.random.rand(1, *img_size, 3).astype(np.float32)

        for key in cfg["models"]:
            path = cfg["models"][key]["save_path"]
            name = cfg["models"][key]["name"]
            if not os.path.isfile(path):
                print(f"{WARN} {name}: model file missing, skipping")
                continue
            model = tf.keras.models.load_model(path)
            out = model.predict(dummy, verbose=0)
            check(f"{name}: output shape {out.shape}, value {out[0][0]:.4f}",
                  out.shape == (1, 1) and 0 <= out[0][0] <= 1)
    except Exception as e:
        print(f"{FAIL} Smoke test failed: {e}")
        errors.append("smoke_test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Also run model smoke test")
    args = parser.parse_args()

    print("\n" + "═" * 50)
    print("  🔍 PIPELINE VERIFICATION")
    print("═" * 50)

    cfg = load_config()
    verify_config(cfg)
    verify_dataset(cfg)
    verify_models(cfg)
    verify_results(cfg)

    if args.full:
        smoke_test(cfg)

    print("\n" + "═" * 50)
    if errors:
        print(f"  ❌ FAILED — {len(errors)} issue(s) found:")
        for e in errors:
            print(f"     • {e}")
        sys.exit(1)
    else:
        print("  ✅ ALL CHECKS PASSED — pipeline is ready")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
