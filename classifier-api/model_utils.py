import os
import io
import json
import logging
import base64
import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image

SUPPRESS_CUDA_WARNINGS = True


def _import_tensorflow_safely():
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 2)
        try:
            import importlib
            tf_mod = importlib.import_module("tensorflow")
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(devnull)
            os.close(old_stderr_fd)
        return tf_mod
    except Exception:
        return None


tf = _import_tensorflow_safely()

import requests

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "classification_model")
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60

MODEL_CANDIDATES = [
    "model_v3.keras",
    "breast_classification_model.keras",
]


def find_model_in_classification_dir() -> Optional[str]:
    for name in MODEL_CANDIDATES:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def load_class_indices() -> Dict[int, str]:
    path = os.path.join(MODEL_DIR, "class_indices.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        m = json.load(f)
    return {int(v): k for k, v in m.items()}


def init_model() -> Tuple[Optional[object], Dict[int, str]]:
    model_path = find_model_in_classification_dir()
    if model_path is None or tf is None:
        return None, {}
    try:
        model = tf.keras.models.load_model(model_path)
        idx_to_name = load_class_indices()
        return model, idx_to_name
    except Exception:
        return None, {}


def preprocess_image_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_with_model(model, x) -> dict:
    preds = model.predict(x)
    probs = preds[0].tolist()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return {
        "pred_idx": pred_idx,
        "probs": probs,
        "confidence": confidence,
        "is_conclusive": confidence >= CONFIDENCE_THRESHOLD,
    }


def proxy_predict(file_bytes: bytes, filename: str, content_type: str, proxy_url: str) -> dict:
    files = {"file": (filename, file_bytes, content_type)}
    resp = requests.post(proxy_url, files=files, timeout=15)
    resp.raise_for_status()
    return resp.json()


# --- GRAD-CAM FUNCTIONS ---

def get_last_conv_layer_info(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                try:
                    if hasattr(inner_layer, 'output') and len(inner_layer.output.shape) == 4:
                        return inner_layer.name, layer.name
                except Exception:
                    continue

    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                return layer.name, None
        except Exception:
            continue

    raise ValueError("Could not find a convolutional layer in the model.")


def make_gradcam_heatmap(img_array, model, pred_index):
    if tf is None:
        return None

    layer_name, nested_model_name = get_last_conv_layer_info(model)

    if nested_model_name:
        base_model = model.get_layer(nested_model_name)

        grad_base_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[base_model.get_layer(layer_name).output, base_model.output]
        )

        with tf.GradientTape() as tape:
            outputs = grad_base_model(img_array)
            conv_outputs, base_outputs = outputs[0], outputs[1]

            if isinstance(conv_outputs, (list, tuple)): conv_outputs = conv_outputs[0]
            if isinstance(base_outputs, (list, tuple)): base_outputs = base_outputs[0]

            tape.watch(conv_outputs)

            x = base_outputs
            found_base = False
            for layer in model.layers:
                if layer.name == nested_model_name:
                    found_base = True
                    continue
                if found_base:
                    try:
                        x = layer(x, training=False)
                    except TypeError:
                        x = layer(x)

            preds = x
            if isinstance(preds, (list, tuple)): preds = preds[0]
            class_channel = preds[:, pred_index]
    else:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            outputs = grad_model(img_array)
            conv_outputs, preds = outputs[0], outputs[1]

            if isinstance(conv_outputs, (list, tuple)): conv_outputs = conv_outputs[0]
            if isinstance(preds, (list, tuple)): preds = preds[0]

            class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ tf.expand_dims(pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def generate_gradcam_base64(img_bytes: bytes, heatmap: np.ndarray) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_arr = np.array(img)

    heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    alpha = 0.5
    superimposed = cv2.addWeighted(colormap, alpha, img_arr, 1 - alpha, 0)

    out_img = Image.fromarray(superimposed)
    buffer = io.BytesIO()
    out_img.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
