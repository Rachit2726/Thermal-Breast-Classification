"""
Unified training script for ResNet50, EfficientNetB0, and MobileNetV2.
Usage:
    python train.py --model resnet
    python train.py --model efficientnet
    python train.py --model mobilenet
"""
import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from config_loader import load_config

ARCHITECTURES = {
    "resnet": (ResNet50, "resnet"),
    "efficientnet": (EfficientNetB0, "efficientnet"),
    "mobilenet": (MobileNetV2, "mobilenet"),
}


def get_class_weights(train_dir, classes):
    labels = []
    for idx, cls in enumerate(classes):
        count = len(os.listdir(os.path.join(train_dir, cls)))
        labels.extend([idx] * count)
    labels = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return {i: w for i, w in enumerate(weights)}


def build_model(arch_fn, img_size, fine_tune_layers):
    base = arch_fn(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base.layers[-fine_tune_layers:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(1, activation="sigmoid")(x)
    return Model(base.input, out)


def train(model_name):
    cfg = load_config()
    t_cfg = cfg["training"]
    img_size = tuple(cfg["image"]["size"])
    data_dir = cfg["data"]["final_dir"]
    save_path = cfg["models"][model_name]["save_path"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(cfg["results"]["log_dir"], exist_ok=True)

    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"), target_size=img_size,
        batch_size=t_cfg["batch_size"], class_mode="binary"
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(data_dir, "val"), target_size=img_size,
        batch_size=t_cfg["batch_size"], class_mode="binary"
    )

    class_weights = get_class_weights(
        os.path.join(data_dir, "train"), cfg["data"]["classes"]
    )
    print(f"Class weights: {class_weights}")

    arch_fn, _ = ARCHITECTURES[model_name]
    model = build_model(arch_fn, img_size, t_cfg["fine_tune_layers"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(t_cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
    )

    callbacks = [
        EarlyStopping(patience=t_cfg["early_stop_patience"], restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True, monitor="val_accuracy"),
        CSVLogger(os.path.join(cfg["results"]["log_dir"], f"{model_name}_training.csv")),
    ]

    model.fit(
        train_data, validation_data=val_data,
        epochs=t_cfg["epochs"], class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(save_path)
    print(f"✅ {cfg['models'][model_name]['name']} saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a breast cancer classifier")
    parser.add_argument("--model", choices=list(ARCHITECTURES.keys()), required=True,
                        help="Model architecture to train")
    args = parser.parse_args()
    train(args.model)
