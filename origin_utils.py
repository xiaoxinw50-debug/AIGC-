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


def aligned_probabilities(model: Any, x: np.ndarray, labels: list[str]) -> np.ndarray:
    raw = multiclass_probs(model, x)
    values = np.asarray([float(raw.get(label, 0.0)) for label in labels], dtype="float64")
    return values / max(float(values.sum()), 1e-12)


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


def matrix_from_features(feats: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray:
    names = list(bundle["feature_names"])
    x = np.asarray([[float(feats[name]) for name in names]], dtype="float32")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fuse_platform_probabilities(
    native: np.ndarray,
    robust: np.ndarray,
    *,
    native_weight: float,
    temperature: float,
) -> np.ndarray:
    combined = native_weight * native + (1.0 - native_weight) * robust
    logits = np.log(np.clip(combined, 1e-9, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def distribution_drift(feats: dict[str, Any], config: dict[str, Any]) -> tuple[float, bool]:
    reference = config.get("drift_reference") or {}
    names = list(reference.get("feature_names") or [])
    if not names:
        return 0.0, False
    median = reference.get("median") or {}
    iqr = reference.get("iqr") or {}
    distances = []
    for name in names:
        scale = max(float(iqr.get(name, 1.0)), 1e-6)
        distance = abs(float(feats.get(name, 0.0)) - float(median.get(name, 0.0))) / scale
        distances.append(min(distance, 20.0))
    score = float(np.mean(distances)) if distances else 0.0
    threshold = float(reference.get("threshold", float("inf")))
    return score, score > threshold


def predict_one(
    image_path: Path,
    binary_bundle: dict[str, Any],
    platform_bundle: dict[str, Any],
    robust_platform_bundle: dict[str, Any] | None = None,
    platform_open_set_config: dict[str, Any] | None = None,
    *,
    generated_threshold: float = 0.75,
    real_threshold: float = 0.35,
) -> dict[str, Any]:
    x_binary, feats = matrix_for_bundle(image_path, binary_bundle)
    x_platform = matrix_from_features(feats, platform_bundle)

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
    config = platform_open_set_config or {}
    labels = list(config.get("platform_labels") or platform_bundle.get("platform_labels") or [])
    if not labels:
        labels = [str(label) for label in platform_model.classes_]
    native_probs = aligned_probabilities(platform_model, x_platform, labels)
    if robust_platform_bundle is not None:
        robust_x = matrix_from_features(feats, robust_platform_bundle)
        robust_probs = aligned_probabilities(robust_platform_bundle["model"], robust_x, labels)
    else:
        robust_probs = native_probs.copy()

    native_weight = float(config.get("native_weight", 1.0))
    temperature = float(config.get("temperature", 1.0))
    fused_probs = fuse_platform_probabilities(
        native_probs,
        robust_probs,
        native_weight=native_weight,
        temperature=temperature,
    )
    order = np.argsort(fused_probs)
    top1_index = int(order[-1])
    top2_index = int(order[-2]) if len(order) > 1 else top1_index
    platform_pred = labels[top1_index]
    platform_confidence = float(fused_probs[top1_index])
    platform_margin = float(fused_probs[top1_index] - fused_probs[top2_index])
    native_pred = labels[int(np.argmax(native_probs))]
    robust_pred = labels[int(np.argmax(robust_probs))]
    model_agreement = native_pred == robust_pred
    drift_score, drift_flag = distribution_drift(feats, config)

    min_confidence = float(config.get("min_confidence", 0.0))
    min_margin = float(config.get("min_margin", 0.0))
    require_agreement = bool(config.get("require_model_agreement", False))
    rejection_reasons: list[str] = []
    if platform_confidence < min_confidence:
        rejection_reasons.append("confidence_below_threshold")
    if platform_margin < min_margin:
        rejection_reasons.append("margin_below_threshold")
    if require_agreement and not model_agreement:
        rejection_reasons.append("model_disagreement")
    if drift_flag:
        rejection_reasons.append("feature_distribution_shift")
    platform_accepted = not rejection_reasons
    platform_prob_map = {label: float(prob) for label, prob in zip(labels, fused_probs)}
    native_prob_map = {label: float(prob) for label, prob in zip(labels, native_probs)}
    robust_prob_map = {label: float(prob) for label, prob in zip(labels, robust_probs)}
    if binary_pred == "generated":
        final_pred = platform_pred if platform_accepted else str(config.get("unknown_label", "unknown_platform"))
        risk_level = "high" if platform_accepted else "review"
        risk_text = "高风险核验" if platform_accepted else "人工复核"
    elif binary_pred == "uncertain":
        final_pred = "uncertain"
        risk_level = "review"
        risk_text = "人工复核"
    else:
        final_pred = "real"
        risk_level = "low"
        risk_text = "自动通过"

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
        "pred_platform_native_probabilities": native_prob_map,
        "pred_platform_robust_probabilities": robust_prob_map,
        "platform_native_pred": native_pred,
        "platform_robust_pred": robust_pred,
        "platform_model_agreement": model_agreement,
        "platform_confidence": round(platform_confidence, 8),
        "platform_margin": round(platform_margin, 8),
        "platform_drift_score": round(drift_score, 8),
        "platform_drift_flag": drift_flag,
        "platform_accepted": platform_accepted,
        "platform_rejection_reasons": rejection_reasons,
        "platform_open_set_version": str(config.get("version", "disabled")),
        "platform_min_confidence": min_confidence,
        "platform_min_margin": min_margin,
        "platform_require_agreement": require_agreement,
        "risk_level": risk_level,
        "risk_text": risk_text,
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
            "edge_mean": float(feats.get("edge_mean", 0.0)),
            "laplacian_var": float(feats.get("laplacian_var", 0.0)),
            "high_freq_ratio": float(feats.get("high_freq_ratio", 0.0)),
            "saturation_mean": float(feats.get("saturation_mean", 0.0)),
            "gray_std": float(feats.get("gray_std", 0.0)),
            "file_size_log": float(feats.get("file_size_log", 0.0)),
        },
    }
