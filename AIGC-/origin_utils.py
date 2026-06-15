from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from feature_utils import compute_image_features


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def generated_prob(model: Any, x: np.ndarray, generated_label: str) -> float:
    classes = list(model.classes_)
    idx = classes.index(generated_label)
    return float(model.predict_proba(x)[0][idx])


def multiclass_probs(model: Any, x: np.ndarray) -> dict[str, float]:
    classes = list(model.classes_)
    probs = model.predict_proba(x)[0]
    return {str(label): float(prob) for label, prob in zip(classes, probs)}


def predict_one(image_path: Path, binary_bundle: dict[str, Any], platform_bundle: dict[str, Any]) -> dict[str, Any]:
    feature_mode = str(binary_bundle["feature_mode"])
    feature_names = list(binary_bundle["feature_names"])
    feats = compute_image_features(image_path, mode=feature_mode)
    x = np.asarray([[float(feats[name]) for name in feature_names]], dtype="float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    binary_model = binary_bundle["model"]
    platform_model = platform_bundle["model"]
    generated_label = str(binary_bundle["generated_label"])
    real_label = str(binary_bundle["real_label"])

    binary_pred = str(binary_model.predict(x)[0])
    platform_pred = str(platform_model.predict(x)[0])
    prob_generated = generated_prob(binary_model, x, generated_label)
    platform_prob_map = multiclass_probs(platform_model, x)
    final_pred = real_label if binary_pred == real_label else platform_pred

    return {
        "image_path": str(image_path),
        "feature_mode": feature_mode,
        "pred_binary_label": binary_pred,
        "pred_prob_generated": round(prob_generated, 8),
        "pred_platform_if_generated": platform_pred,
        "pred_platform_probabilities": platform_prob_map,
        "pred_final_label": final_pred,
        "feature_snapshot": {
            "orig_width": float(feats.get("orig_width", 0.0)),
            "orig_height": float(feats.get("orig_height", 0.0)),
            "aspect_ratio": float(feats.get("aspect_ratio", 0.0)),
            "megapixels": float(feats.get("megapixels", 0.0)),
            "file_size_kb": float(feats.get("file_size_kb", 0.0)),
            "has_alpha": float(feats.get("has_alpha", 0.0)),
            "is_png": float(feats.get("is_png", 0.0)),
            "is_jpeg": float(feats.get("is_jpeg", 0.0)),
            "is_webp": float(feats.get("is_webp", 0.0)),
            "exif_present": float(feats.get("exif_present", 0.0)),
            "info_key_count": float(feats.get("info_key_count", 0.0)),
            "band_count": float(feats.get("band_count", 0.0)),
        },
    }
