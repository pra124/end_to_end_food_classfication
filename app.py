# app.py
import os
import io
import json
import logging
import threading
import shutil
from typing import Optional, Tuple, Dict, Any, List

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import requests

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax

# ------------------------------
# Basic app setup & config
# ------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Config (you can override via environment variables)
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
MODELS_BASE_DIR = os.environ.get("MODELS_BASE_DIR", os.path.join(BASE_DIR, "models"))
CUSTOM_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "custom_model")
RESNET_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "resnet")
VGG_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "vgg16")

CUSTOM_JSON_FILE = os.environ.get("CUSTOM_JSON_FILE", os.path.join(BASE_DIR, "model_evaluation_results_custom_model.json"))
RESNET_JSON_FILE = os.environ.get("RESNET_JSON_FILE", os.path.join(BASE_DIR, "model_evaluation_results_resnet.json"))
VGG_JSON_FILE = os.environ.get("VGG_JSON_FILE", os.path.join(BASE_DIR, "model_evaluation_results_vgg.json"))
NUTRITION_FILE = os.environ.get("NUTRITION_FILE", os.path.join(BASE_DIR, "food_json.json"))
FOOD_JSON_DIR = os.path.join(BASE_DIR, "food_json")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
NUTRITION_ADMIN_KEY = os.environ.get("NUTRITION_ADMIN_KEY", "changeme123")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
os.makedirs(RESNET_MODELS_DIR, exist_ok=True)
os.makedirs(VGG_MODELS_DIR, exist_ok=True)
os.makedirs(FOOD_JSON_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Model download URL mapping (maps expected model keys to provided URLs)
# The user provided three groups of URLs earlier; we map them by index.
# ------------------------------
MODEL_DOWNLOAD_URLS = {
    # custom - map custom_modelf1..custom_modelf11 to provided custom_mode1_1..custom_mode1_11
    "custom_modelf1": "https://drive.google.com/uc?export=download&id=19egYnkV8VsTlF_yjYUDArX12xS4bC6sP",
    "custom_modelf2": "https://drive.google.com/uc?export=download&id=1qDG2mKGDvO6yTBjHTLMmyEYa4bsMj8mk",
    "custom_modelf3": "https://drive.google.com/uc?export=download&id=1YmZAiOm5jfekCwmUaIg5H7-Ujen6FTA5",
    "custom_modelf4": "https://drive.google.com/uc?export=download&id=17N_Kg0BVflCcUxmRoclTs2U5i2PlqPQf",
    "custom_modelf5": "https://drive.google.com/uc?export=download&id=1uNqtL37dmfTzgCVK_Soyh8kpULaNGEDu",
    "custom_modelf6": "https://drive.google.com/uc?export=download&id=1RlgT7F5wL_ZYwpzVJiVyCoCZhfbLf66w",
    "custom_modelf7": "https://drive.google.com/uc?export=download&id=1quio-27otIjT9MbrvVHZpm3zqCeRESKd",
    "custom_modelf8": "https://drive.google.com/uc?export=download&id=1vjoYQFdVmS_Jdgeiocko2ecFLHr1QHtC",
    "custom_modelf9": "https://drive.google.com/uc?export=download&id=1JD8C7AO4pxhy4fvNNcNMIDaZ5ZMIbzNw",
    "custom_modelf10":"https://drive.google.com/uc?export=download&id=1JlPtSLpBj8DEd1ZQzrR3cAKPpJOJGCT4",
    "custom_modelf11":"https://drive.google.com/uc?export=download&id=1sr4-l2o_8UKG5cHgkz9QgVPBI3kEdQB0",

    # resnet - map resnetf1..resnetf11 to provided resnet_model_1..11
    "resnetf1": "https://drive.google.com/uc?export=download&id=1n53ruJJG_RlZtsqko2UI7wK51iQx1Kyq",
    "resnetf2": "https://drive.google.com/uc?export=download&id=1iC6Ds6GJcYhakH_kSVO5YdEnkiv9plW3",
    "resnetf3": "https://drive.google.com/uc?export=download&id=1DPDIBVOaXFjXrc0i7RwwtSqQvGtLGt9U",
    "resnetf4": "https://drive.google.com/uc?export=download&id=1sWqutou_nD0W8lUNLe9fD_1wTdQxFcOS",
    "resnetf5": "https://drive.google.com/uc?export=download&id=1YfcYy_W8OpCvUhtVxSwbepMn5nFRL3Vl",
    "resnetf6": "https://drive.google.com/uc?export=download&id=1WaY6nmtdNZ_M8maVxy34fdNw5Q8_X7HR",
    "resnetf7": "https://drive.google.com/uc?export=download&id=1WYb54yaO_8l72v25XWDdGs-d-ozI2gsA",
    "resnetf8": "https://drive.google.com/uc?export=download&id=1Lm8eH65ldTS_OWVls4ZKSLC12FfTlnUL",
    "resnetf9": "https://drive.google.com/uc?export=download&id=1VXj549HyOLesgerzxYkj97UO5GgA18X3",
    "resnetf10":"https://drive.google.com/uc?export=download&id=1Up0e450n3a4sMnmf9Kusd8Sm3ahqwBGL",
    "resnetf11":"https://drive.google.com/uc?export=download&id=1uPDfQs76vZdjYnN8CeSHgR2rX-8LCaUE",

    # vgg - map vggf1..vggf11 to provided vgg_model_1..11
    "vggf1":  "https://drive.google.com/uc?export=download&id=1scURjqWGN20K8Gl-JalwrMRoQVQJUzRc",
    "vggf2":  "https://drive.google.com/uc?export=download&id=1mvwXUk1yl-M-ZVx7SJHI-95F0nSfc10z",
    "vggf3":  "https://drive.google.com/uc?export=download&id=1nX2MyQldeghBFlee4ND4AVnT1m4qtaHO",
    "vggf4":  "https://drive.google.com/uc?export=download&id=1QNihazmn7aTIIFyjLIdx3KIeLSWShgxk",
    "vggf5":  "https://drive.google.com/uc?export=download&id=1HM1zUxGbM3maJCwKl8lejpeAmnYLeRcu",
    "vggf6":  "https://drive.google.com/uc?export=download&id=1GNKnyww4A9FWhW_2F9pU7KQ0FS1UVd8Z",
    "vggf7":  "https://drive.google.com/uc?export=download&id=17HfpNHKtOGyhMtb8kgBGY_N08z2ffDy1",
    "vggf8":  "https://drive.google.com/uc?export=download&id=1hgUEMnaKO5OW8BXxsoB949ycvMtly4H9",
    "vggf9":  "https://drive.google.com/uc?export=download&id=1G_5Taziq2wduJW86xYkj97UO5GgA18X3",
    "vggf10": "https://drive.google.com/uc?export=download&id=12Jib9H7EwMspBs9pfD85O_gryhjt67_b",
    "vggf11": "https://drive.google.com/uc?export=download&id=1v_JIE5jkPDPzQz2jz3buxlrBFYxapvR7",
}

# ------------------------------
# Helper utilities
# ------------------------------
def normalize_for_key(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def titleize_spaces(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.replace("_", " ").title()

def allowed_file_extension(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def safe_json_load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning("Failed to read JSON %s: %s", path, e)
    return {}

def safe_json_write(path: str, data: dict) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.exception("Failed to write JSON %s: %s", path, e)
        return False

# ------------------------------
# Fallback classes (keeps dropdown working if nutrition JSON missing)
# ------------------------------
FALLBACK_CLASSES = [
    'apple_pie','baked_potato','burger','butter_naan','chai','chapati',
    'cheesecake','chicken_curry','chole_bhature','crispy_chicken','dal_makhani',
    'dhokla','donut','fried_rice','fries','hot_dog','ice_cream','idli','jalebi',
    'kaathi_rolls','kadai_paneer','kulfi','masala_dosa','momos','omelette',
    'paani_puri','pakode','pav_bhaji','pizza','samosa','sandwich','sushi','taco'
]

def build_class_list_from_nutrition() -> list:
    data = safe_json_load(NUTRITION_FILE)
    if data:
        return sorted(list(data.keys()))
    return FALLBACK_CLASSES

# ------------------------------
# Simple model class index (example mapping)
# This was present in your earlier file. Keep as-is (lowercase keys).
# ------------------------------
MODEL_CLASS_INDEX = {
    "custom_modelf1":  {'cheesecake': 0, 'sushi': 1, 'kaathi_rolls': 2},
    "custom_modelf2":  {'ice_cream': 0, 'fries': 1, 'donut': 2},
    "custom_modelf3":  {'dal_makhani': 0, 'chapati': 1, 'pakode': 2},
    "custom_modelf4":  {'baked_potato': 0, 'kadai_paneer': 1, 'dhokla': 2},
    "custom_modelf5":  {'jalebi': 0, 'momos': 1, 'masala_dosa': 2},
    "custom_modelf6":  {'chole_bhature': 0, 'pav_bhaji': 1, 'hot_dog': 2},
    "custom_modelf7":  {'chai': 0, 'chicken_curry': 1, 'taquito': 2},
    "custom_modelf8":  {'crispy_chicken': 0, 'butter_naan': 1, 'fried_rice': 2},
    "custom_modelf9":  {'apple_pie': 0, 'samosa': 1, 'pizza': 2},
    "custom_modelf10": {'kulfi': 0, 'omelette': 1, 'sandwich': 2},
    "custom_modelf11": {'paani_puri': 0, 'taco': 1, 'idli': 2, 'burger': 3},

    "resnetf1":  {'cheesecake': 0, 'sushi': 1, 'kaathi_rolls': 2},
    "resnetf2":  {'ice_cream': 0, 'fries': 1, 'donut': 2},
    "resnetf3":  {'dal_makhani': 0, 'chapati': 1, 'pakode': 2},
    "resnetf4":  {'baked_potato': 0, 'kadai_paneer': 1, 'dhokla': 2},
    "resnetf5":  {'jalebi': 0, 'momos': 1, 'masala_dosa': 2},
    "resnetf6":  {'chole_bhature': 0, 'pav_bhaji': 1, 'hot_dog': 2},
    "resnetf7":  {'chai': 0, 'chicken_curry': 1, 'taquito': 2},
    "resnetf8":  {'crispy_chicken': 0, 'butter_naan': 1, 'fried_rice': 2},
    "resnetf9":  {'apple_pie': 0, 'samosa': 1, 'pizza': 2},
    "resnetf10": {'kulfi': 0, 'omelette': 1, 'sandwich': 2},
    "resnetf11": {'paani_puri': 0, 'taco': 1, 'idli': 2, 'burger': 3},

    "vggf1":  {'cheesecake': 0, 'sushi': 1, 'kaathi_rolls': 2},
    "vggf2":  {'ice_cream': 0, 'fries': 1, 'donut': 2},
    "vggf3":  {'dal_makhani': 0, 'chapati': 1, 'pakode': 2},
    "vggf4":  {'baked_potato': 0, 'kadai_paneer': 1, 'dhokla': 2},
    "vggf5":  {'jalebi': 0, 'momos': 1, 'masala_dosa': 2},
    "vggf6":  {'chole_bhature': 0, 'pav_bhaji': 1, 'hot_dog': 2},
    "vggf7":  {'chai': 0, 'chicken_curry': 1, 'taquito': 2},
    "vggf8":  {'crispy_chicken': 0, 'butter_naan': 1, 'fried_rice': 2},
    "vggf9":  {'apple_pie': 0, 'samosa': 1, 'pizza': 2},
    "vggf10": {'kulfi': 0, 'omelette': 1, 'sandwich': 2},
    "vggf11": {'paani_puri': 0, 'taco': 1, 'idli': 2, 'burger': 3},
}

# ------------------------------
# Download helpers
# ------------------------------
def ensure_single_file_in_folder(folder: str, keep_filename: Optional[str]):
    """
    Remove all files in folder except keep_filename (if provided).
    """
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        full = os.path.join(folder, fname)
        if fname == keep_filename:
            continue
        try:
            if os.path.isfile(full) or os.path.islink(full):
                os.remove(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
        except Exception:
            logger.exception("Failed to remove %s", full)

def download_file_stream(url: str, dest_path: str, chunk_size=8192, timeout=60):
    """
    Download file via streaming. This supports direct links (Google drive links may need special handling).
    """
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        # If Google drive returns HTML page rather than file, user must provide direct downloadable link.
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        logger.exception("Failed to download %s -> %s", url, dest_path)
        return False

def get_model_folder_and_filename(model_key: str, model_group: str) -> Tuple[str, str]:
    """
    Return (folder_path, filename) for storing the model.
    We'll use <models_base>/<model_group>/<model_key>/<model_key>.h5
    """
    group_dir = {
        "custom_model": CUSTOM_MODELS_DIR,
        "resnet": RESNET_MODELS_DIR,
        "vgg16": VGG_MODELS_DIR
    }.get(model_group, custom_safe(CUSTOM_MODELS_DIR))

    folder = os.path.join(group_dir, model_key)
    os.makedirs(folder, exist_ok=True)
    filename = model_key + ".h5"
    return folder, filename

def custom_safe(x):
    return x

def download_model_if_missing(model_key: str, model_group: str) -> Optional[str]:
    """
    Ensure the model .h5 exists on disk. If not present, try to download using MODEL_DOWNLOAD_URLS.
    Returns full path to the model .h5 or None on failure.
    """
    if not model_key:
        return None

    folder, filename = get_model_folder_and_filename(model_key, model_group)
    dest_path = os.path.join(folder, filename)

    # If file exists, make sure folder contains only that file and return
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        ensure_single_file_in_folder(folder, filename)
        return dest_path

    # If mapping provided, attempt to download
    url = MODEL_DOWNLOAD_URLS.get(model_key)
    if not url:
        logger.warning("No download URL for model key %s", model_key)
        return None

    # If any files exist in folder, delete them to maintain single-file requirement
    ensure_single_file_in_folder(folder, None)

    # Download to a temp path then move
    tmp_path = os.path.join(folder, filename + ".part")
    ok = download_file_stream(url, tmp_path)
    if not ok:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
        return None

    try:
        os.rename(tmp_path, dest_path)
    except Exception:
        # fallback: copy
        shutil.copy2(tmp_path, dest_path)
        try: os.remove(tmp_path)
        except Exception: pass

    ensure_single_file_in_folder(folder, filename)
    if os.path.exists(dest_path):
        return dest_path
    return None

# ------------------------------
# Model caching & loader
# ------------------------------
_model_cache: Dict[str, Any] = {}
_model_cache_lock = threading.Lock()

def load_model_cached(path: str):
    if not path:
        raise ValueError("No model path provided")
    with _model_cache_lock:
        if path in _model_cache:
            return _model_cache[path]
    try:
        model = load_model(path, compile=False)
    except Exception:
        logger.exception("Initial model load failed for %s. Clearing TF session and retrying.", path)
        try:
            tf.keras.backend.clear_session()
        except Exception:
            logger.debug("Failed clearing TF session (ignored).")
        model = load_model(path, compile=False)
    with _model_cache_lock:
        _model_cache[path] = model
    return model

# ------------------------------
# JSON helpers & model selection helpers
# ------------------------------
def find_model_from_json(eval_json: dict, classname: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Try to extract model_used entry for a classname from an evaluation JSON dict.
    """
    if not eval_json or not classname:
        return None, None
    variants = [classname, classname.lower(), normalize_for_key(classname), titleize_spaces(classname)]
    seen = set()
    for v in variants:
        if not v or v in seen:
            continue
        seen.add(v)
        if v in eval_json:
            entry = eval_json[v]
            mu = entry.get("model_used") or entry.get("model") or entry.get("model_used_name")
            return mu, entry
    # try normalized keys
    target = normalize_for_key(classname)
    for key, entry in eval_json.items():
        if normalize_for_key(key) == target:
            mu = entry.get("model_used") or entry.get("model") or entry.get("model_used_name")
            return mu, entry
    return None, None

def load_json_file(path: str) -> dict:
    return safe_json_load(path)

def get_model_path_from_model_used(model_used: str, model_group: str) -> Optional[str]:
    """
    Resolve model path inside model_group folder (case-insensitive match).
    If missing, attempt to download via MODEL_DOWNLOAD_URLS.
    """
    if not model_used:
        return None
    base = os.path.splitext(str(model_used))[0]
    base_lower = base.lower()

    if model_group == "custom_model":
        folder = CUSTOM_MODELS_DIR
    elif model_group == "resnet":
        folder = RESNET_MODELS_DIR
    elif model_group == "vgg16":
        folder = VGG_MODELS_DIR
    else:
        folder = CUSTOM_MODELS_DIR

    # look for existing .h5
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            candidate = os.path.join(folder, fn)
            if os.path.isdir(candidate):
                # search inside subfolder <model_key>/<model_key>.h5
                possible = os.path.join(candidate, base + ".h5")
                if os.path.exists(possible):
                    return possible
            else:
                # top-level file (not expected), try match
                if fn.lower().startswith(base_lower) and fn.lower().endswith(".h5"):
                    return os.path.join(folder, fn)

    # try mapping model_used exactly as key (lowercase)
    model_key = base_lower
    # ensure consistent key naming: in our mapping the keys are like "custom_modelf1"
    # try to use provided model_used as-is, or with 'f' variants
    model_key_candidates = [base, base_lower]
    # also try some common substitutions:
    model_key_candidates += [base_lower.replace("modelf", "modelf"), base_lower.replace("modelf", "modelf")]
    # pick first that exists in mapping
    chosen_key = None
    for k in model_key_candidates:
        if k in MODEL_DOWNLOAD_URLS:
            chosen_key = k
            break
    if not chosen_key:
        # last attempt: try model_used exactly if it matches mapping
        if model_used in MODEL_DOWNLOAD_URLS:
            chosen_key = model_used

    if not chosen_key:
        # cannot resolve via mapping
        return None

    # try to ensure model is downloaded
    model_path = download_model_if_missing(chosen_key, model_group)
    return model_path

# ------------------------------
# Preprocess image (Pillow)
# ------------------------------
def preprocess_dynamic(img: Image.Image, w: int, h: int):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_resized = img.resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        img_rgb = img_resized.convert("RGB")
        arr = np.asarray(img_rgb).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def get_class_from_index(model_key_base: str, idx: int) -> Optional[str]:
    mapping = MODEL_CLASS_INDEX.get(model_key_base, {})
    for c, v in mapping.items():
        if v == idx:
            return c
    return None

# ------------------------------
# Nutrition loader
# ------------------------------
def NutritionLoader(class_name):
    if not class_name:
        return {}
    norm = normalize_for_key(class_name)
    file_path = os.path.join(FOOD_JSON_DIR, norm + ".json")
    if not os.path.exists(file_path):
        return {
            "calories": "N/A",
            "protein": "N/A",
            "fat": "N/A",
            "carbohydrates": "N/A",
            "fiber": "N/A"
        }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "calories": "N/A",
            "protein": "N/A",
            "fat": "N/A",
            "carbohydrates": "N/A",
            "fiber": "N/A"
        }

# ------------------------------
# Classification report builder (from confusion matrix or metrics)
# ------------------------------
def build_classification_report_from_confusion(conf_matrix: List[List[int]], labels: Optional[List[str]]=None) -> dict:
    """
    conf_matrix: nested lists where conf_matrix[i][j] = true class i predicted as j
    labels: optional list of label names aligned with classes
    Returns dict similar to sklearn's classification_report structure.
    """
    try:
        cm = np.array(conf_matrix, dtype=np.int64)
        if cm.ndim != 2:
            raise ValueError("confusion matrix must be 2D")
        n_classes = cm.shape[0]
        # if rectangular, use min dimension for classes
        # compute per-class metrics
        precision = []
        recall = []
        f1 = []
        support = []
        for i in range(n_classes):
            tp = int(cm[i, i]) if i < cm.shape[1] else 0
            fn = int(cm[i, :].sum()) - tp
            fp = int(cm[:, i].sum()) - tp
            tn = int(cm.sum()) - (tp + fp + fn)
            sup = tp + fn
            if sup == 0:
                rec = 0.0
            else:
                rec = tp / sup
            denom = tp + fp
            if denom == 0:
                prec = 0.0
            else:
                prec = tp / denom
            if prec + rec == 0:
                f1s = 0.0
            else:
                f1s = 2 * (prec * rec) / (prec + rec)
            precision.append(round(prec, 4))
            recall.append(round(rec, 4))
            f1.append(round(f1s, 4))
            support.append(int(sup))
        report = {}
        for idx in range(n_classes):
            name = labels[idx] if labels and idx < len(labels) else str(idx)
            report[name] = {
                "precision": precision[idx],
                "recall": recall[idx],
                "f1-score": f1[idx],
                "support": support[idx]
            }
        # add macro avg, weighted avg, accuracy
        total_support = sum(support)
        accuracy = float(cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
        macro_prec = float(np.mean(precision)) if precision else 0.0
        macro_rec = float(np.mean(recall)) if recall else 0.0
        macro_f1 = float(np.mean(f1)) if f1 else 0.0
        weighted_prec = float(np.average(precision, weights=support)) if total_support > 0 else 0.0
        weighted_rec = float(np.average(recall, weights=support)) if total_support > 0 else 0.0
        weighted_f1 = float(np.average(f1, weights=support)) if total_support > 0 else 0.0
        report["accuracy"] = round(accuracy, 4)
        report["macro avg"] = {"precision": round(macro_prec, 4), "recall": round(macro_rec, 4), "f1-score": round(macro_f1, 4), "support": total_support}
        report["weighted avg"] = {"precision": round(weighted_prec, 4), "recall": round(weighted_rec, 4), "f1-score": round(weighted_f1, 4), "support": total_support}
        return report
    except Exception:
        logger.exception("Failed to build classification report from confusion")
        return {}

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def index():
    classes = build_class_list_from_nutrition()
    try:
        return render_template("index.html", classes=classes)
    except Exception:
        return jsonify({"classes": classes})

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)

@app.route("/get_nutrition")
def get_nutrition():
    cls = request.args.get("class", "")
    return jsonify(NutritionLoader(cls))

@app.route("/update_nutrition", methods=["POST"])
def update_nutrition():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400
    admin_key = body.get("admin_key")
    if not admin_key or admin_key != NUTRITION_ADMIN_KEY:
        return jsonify({"success": False, "error": "Invalid admin key"}), 403
    class_key = body.get("class_key")
    nutrition_values = body.get("nutrition")
    replace = body.get("replace", True)
    if not class_key or not isinstance(nutrition_values, dict):
        return jsonify({"success": False, "error": "class_key and nutrition dict required"}), 400
    store_name = normalize_for_key(class_key)
    try:
        existing = safe_json_load(NUTRITION_FILE) or {}
        if replace:
            existing[store_name] = nutrition_values
        else:
            existing.setdefault(store_name, {}).update(nutrition_values)
        if not safe_json_write(NUTRITION_FILE, existing):
            raise RuntimeError("Failed to persist nutrition file")
    except Exception as e:
        logger.exception("Failed to write nutrition file: %s", e)
        return jsonify({"success": False, "error": f"Failed to write file: {str(e)}"}), 500
    return jsonify({"success": True, "stored": store_name, "nutrition": nutrition_values})

@app.route("/predict", methods=["POST"])
def predict():
    # Validate file
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "Missing file"}), 400
    filename = secure_filename(file.filename)
    if not allowed_file_extension(filename):
        return jsonify({"success": False, "error": "Unsupported file type"}), 400
    # Save upload (but ensure uploads folder contains only this file)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        # remove any existing uploads (to keep only one)
        for f in os.listdir(UPLOAD_FOLDER):
            try:
                fp = os.path.join(UPLOAD_FOLDER, f)
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception:
                logger.exception("Failed to remove old upload %s", f)
        file.save(filepath)
    except Exception as e:
        logger.exception("Failed to save uploaded file: %s", e)
        return jsonify({"success": False, "error": "Failed to save uploaded file"}), 500
    # Read form data
    selected_class = (request.form.get("selected_class")
                      or request.form.get("class")
                      or request.form.get("selectedClass")
                      or request.form.get("class_name"))
    model_type = (request.form.get("model_type")
                  or request.form.get("modelType")
                  or request.form.get("model"))
    if not selected_class:
        return jsonify({"success": False, "error": "Selected class missing"}), 400
    if not model_type:
        return jsonify({"success": False, "error": "Model type missing"}), 400
    mt = model_type.lower()
    if "custom" in mt:
        model_group = "custom_model"
        eval_json = load_json_file(CUSTOM_JSON_FILE)
    elif "resnet" in mt:
        model_group = "resnet"
        eval_json = load_json_file(RESNET_JSON_FILE)
    elif "vgg" in mt:
        model_group = "vgg16"
        eval_json = load_json_file(VGG_JSON_FILE)
    else:
        model_group = "custom_model"
        eval_json = load_json_file(CUSTOM_JSON_FILE)
    # Find model_used in evaluation JSON
    model_used, class_entry = find_model_from_json(eval_json, selected_class)
    # fallback: try MODEL_CLASS_INDEX keys for presence of selected_class
    if not model_used:
        sel_norm = normalize_for_key(selected_class)
        for m_key, label_map in MODEL_CLASS_INDEX.items():
            if sel_norm in label_map:
                model_used = m_key
                break
    if not model_used:
        return jsonify({"success": False, "error": "Model not found for this class"}), 400
    # Resolve model file on disk; attempt download if missing
    model_path = get_model_path_from_model_used(model_used, model_group)
    if not model_path:
        logger.error("Model path could not be resolved for model_used=%s, group=%s", model_used, model_group)
        return jsonify({"success": False, "error": "Model file missing on disk (and download failed)"}), 500
    # Load model
    try:
        model = load_model_cached(model_path)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return jsonify({"success": False, "error": f"Failed to load model: {str(e)}"}), 500
    # Determine input size
    try:
        shape = model.input_shape
        if isinstance(shape, tuple) and len(shape) == 4:
            _, h, w, _ = shape
            if h is None or w is None:
                w, h = 224, 224
        else:
            w, h = 224, 224
    except Exception:
        w, h = 224, 224
    # Open image and preprocess
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as e:
        logger.exception("Failed to open image: %s", e)
        return jsonify({"success": False, "error": f"Failed to open image: {str(e)}"}), 400
    x = preprocess_dynamic(img, w, h)
    # Predict
    try:
        pred = model.predict(x)
        pred = np.array(pred).squeeze()
    except Exception as e:
        logger.exception("Model prediction failed: %s", e)
        return jsonify({"success": False, "error": f"Model prediction failed: {str(e)}"}), 500
    # Normalize probabilities if needed
    try:
        if pred.ndim == 0:
            pred = np.array([pred])
        if np.any(pred < 0) or (not np.isclose(np.sum(pred), 1.0) and np.sum(pred) != 0):
            pred = softmax(pred).numpy()
    except Exception:
        pred = np.array(pred)
    # Pick top index + confidence
    try:
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
    except Exception:
        idx = 0
        conf = float(pred[0]) if isinstance(pred, (list, np.ndarray)) and len(pred) > 0 else 0.0
    # Map predicted index to label
    model_used_base = os.path.splitext(str(model_used))[0].lower()
    pred_label = get_class_from_index(model_used_base, idx)
    if not pred_label:
        mapping = MODEL_CLASS_INDEX.get(model_used_base, {})
        for k, v in mapping.items():
            if v == idx:
                pred_label = k
                break
    pred_label = pred_label or normalize_for_key(selected_class)
    # Attempt to collect metrics and confusion
    metrics = {}
    confusion = None
    classification_report = None
    if class_entry and isinstance(class_entry, dict):
        # baseline metric keys
        for k in ["accuracy", "precision", "recall", "f1_score", "true_positive", "false_positive", "false_negative", "true_negative", "tp", "fp", "tn", "fn"]:
            if k in class_entry:
                metrics[k] = class_entry.get(k)
        confusion = class_entry.get("confusion_matrix_full") or class_entry.get("confusion_matrix") or class_entry.get("confusion")
    else:
        # search across evaluation JSON files
        for path in (CUSTOM_JSON_FILE, RESNET_JSON_FILE, VGG_JSON_FILE):
            eval_j = load_json_file(path)
            m_used, c_entry = find_model_from_json(eval_j, selected_class)
            if c_entry:
                for k in ["accuracy", "precision", "recall", "f1_score", "true_positive", "false_positive", "false_negative", "true_negative"]:
                    if k in c_entry:
                        metrics[k] = c_entry.get(k)
                confusion = c_entry.get("confusion_matrix_full") or c_entry.get("confusion_matrix")
                break
    # build classification report if confusion matrix found
    if confusion and isinstance(confusion, (list, list)):
        try:
            labels = None
            # if MODEL_CLASS_INDEX available for model_used_base, use those keys as labels
            mapping = MODEL_CLASS_INDEX.get(model_used_base)
            if mapping:
                # order labels by index
                labels = [None] * (max(mapping.values()) + 1)
                for kname, idxnum in mapping.items():
                    if idxnum < len(labels):
                        labels[idxnum] = kname
            classification_report = build_classification_report_from_confusion(confusion, labels)
        except Exception:
            logger.exception("Failed to build classification report")
            classification_report = None
    elif metrics:
        # if no confusion but we have metrics, return these
        classification_report = {"summary_metrics": metrics}
    else:
        classification_report = None
    # nutrition info
    nutrition_selected = NutritionLoader(selected_class)
    nutrition_predicted = NutritionLoader(pred_label)
    raw_probabilities = np.array(pred).tolist() if pred is not None else []
    response = {
        "success": True,
        "selected_class": selected_class,
        "predicted_label": pred_label,
        "confidence": conf,
        "model_used": model_used,
        "model_type": model_group,
        "metrics": metrics if metrics else None,
        "confusion_matrix": confusion if confusion else None,
        "classification_report": classification_report,
        "nutrition_selected": nutrition_selected,
        "nutrition_predicted": nutrition_predicted,
        "raw_probabilities": raw_probabilities
    }
    if app.debug:
        response["_debug"] = {"model_path": model_path}
    else:
        logger.info("Prediction complete - class=%s predicted=%s conf=%.4f model=%s", selected_class, pred_label, conf, model_used)
    return jsonify(response)

# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug_mode)