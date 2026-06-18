from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from feature_utils import compute_image_features


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def generated_prob(model: Any, x: np.ndarray, generated_label: Any) -> float:
    classes = list(model.classes_)
    if generated_label in classes:
        idx = classes.index(generated_label)
    else:
        idx = [str(label) for label in classes].index(str(generated_label))
    return float(model.predict_proba(x)[0][idx])


def multiclass_probs(model: Any, x: np.ndarray) -> dict[str, float]:
    classes = list(model.classes_)
    probs = model.predict_proba(x)[0]
    return {str(label): float(prob) for label, prob in zip(classes, probs)}


def normalize_binary_label(value: Any, real_label: Any, generated_label: Any) -> str:
    if value == generated_label or str(value) == str(generated_label):
        return "generated"
    if value == real_label or str(value) == str(real_label):
        return "real"
    return str(value)


def matrix_for_bundle(image_path: Path, bundle: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    feature_mode = str(bundle["feature_mode"])
    names = list(bundle["feature_names"])
    feats = compute_image_features(image_path, mode=feature_mode)
    x = np.asarray([[float(feats[name]) for name in names]], dtype="float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, feats


def predict_one(
    image_path: Path,
    binary_bundle: dict[str, Any],
    platform_bundle: dict[str, Any],
    *,
    generated_threshold: float = 0.75,
    real_threshold: float = 0.35,
) -> dict[str, Any]:
    x_binary, feats = matrix_for_bundle(image_path, binary_bundle)
    x_platform, _ = matrix_for_bundle(image_path, platform_bundle)

    binary_model = binary_bundle["model"]
    platform_model = platform_bundle["model"]
    generated_label = binary_bundle["generated_label"]
    real_label = binary_bundle["real_label"]

    raw_binary_pred = binary_model.predict(x_binary)[0]
    prob_generated = generated_prob(binary_model, x_binary, generated_label)
    if prob_generated >= generated_threshold:
        binary_pred = "generated"
        decision_status = "generated"
        decision_text = "AI 生成"
    elif prob_generated <= real_threshold:
        binary_pred = "real"
        decision_status = "real"
        decision_text = "真实图片"
    else:
        binary_pred = "uncertain"
        decision_status = "uncertain"
        decision_text = "需人工复核"

    raw_model_label = normalize_binary_label(raw_binary_pred, real_label, generated_label)
    platform_pred = str(platform_model.predict(x_platform)[0])
    platform_prob_map = multiclass_probs(platform_model, x_platform)
    final_pred = platform_pred if binary_pred == "generated" else "real"

    return {
        "image_path": str(image_path),
        "feature_mode": str(binary_bundle["feature_mode"]),
        "binary_model_raw_label": raw_model_label,
        "decision_status": decision_status,
        "decision_text": decision_text,
        "generated_threshold": generated_threshold,
        "real_threshold": real_threshold,
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
