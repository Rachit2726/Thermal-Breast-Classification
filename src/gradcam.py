"""
GradCAM heatmap generator for model explainability.
"""
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(model, img_array, class_idx=1):
    last_conv = get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0] if class_idx == 1 else 1 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_resized = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = np.uint8(img_array[0] * 255)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay), Image.fromarray(heatmap)
