import os
import io
import json
import logging
import base64

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
