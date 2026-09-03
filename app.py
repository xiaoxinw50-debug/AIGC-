from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model_payload import ensure_model_artifacts
from origin_utils import load_bundle, predict_one


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ensure_model_artifacts(ROOT)
BINARY_BUNDLE = MODEL_DIR / "binary_model_bundle.joblib"
PLATFORM_BUNDLE = MODEL_DIR / "platform_model_bundle.joblib"
BINARY_TOP = MODEL_DIR / "binary_top_features.csv"
PLATFORM_TOP = MODEL_DIR / "platform_top_features.csv"
THRESHOLD_PROFILES_CSV = MODEL_DIR / "web_threshold_profiles_v4.csv"

app = Flask(__name__, template_folder=str(ROOT / "webapp" / "templates"), static_folder=str(ROOT / "webapp" / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

binary_bundle = load_bundle(BINARY_BUNDLE)
platform_bundle = load_bundle(PLATFORM_BUNDLE)

PROFILE_UI_NOTES = {
    "operational_low_false_ai": "默认展示档：优先避免真实图被误判为 AI，复核负担最低。",
    "balanced_review": "均衡复核档：适合课程演示和平台审核，把更多边界样本交给人工复核。",
    "high_risk_after_sales_review": "售后高风险档：适合异物、瑕疵、仅退款纠纷线索初筛，宁可多复核也不轻易放过可疑图。",
}


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_policy_profiles() -> list[dict[str, float | str]]:
    profiles: list[dict[str, float | str]] = []
    if THRESHOLD_PROFILES_CSV.exists():
        with THRESHOLD_PROFILES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                profile_id = row.get("profile_id", "")
                profiles.append(
                    {
                        "profile_id": profile_id,
                        "profile_name": row.get("profile_name", profile_id),
                        "selection_note": row.get("selection_note", ""),
                        "ui_note": PROFILE_UI_NOTES.get(profile_id, "当前模型阈值策略。"),
                        "real_threshold": _float_value(row, "real_threshold", 0.45),
                        "generated_threshold": _float_value(row, "generated_threshold", 0.75),
                        "test_real_false_ai_rate": _float_value(row, "test_real_false_ai_rate"),
                        "test_generated_auto_recall": _float_value(row, "test_generated_auto_recall"),
                        "test_generated_risk_capture_rate": _float_value(row, "test_generated_risk_capture_rate"),
                        "test_review_rate": _float_value(row, "test_review_rate"),
                    }
                )
    if not profiles:
        profiles.append(
            {
                "profile_id": "operational_low_false_ai",
                "profile_name": "运营低误伤",
                "selection_note": "fallback",
                "ui_note": PROFILE_UI_NOTES["operational_low_false_ai"],
                "real_threshold": float(binary_bundle.get("recommended_real_threshold", 0.45)),
                "generated_threshold": float(binary_bundle.get("recommended_generated_threshold", 0.75)),
                "test_real_false_ai_rate": 0.0,
                "test_generated_auto_recall": 0.0,
                "test_generated_risk_capture_rate": 0.0,
                "test_review_rate": 0.0,
            }
        )
    return profiles


POLICY_PROFILES = load_policy_profiles()
POLICY_BY_ID = {str(profile["profile_id"]): profile for profile in POLICY_PROFILES}
DEFAULT_POLICY_ID = os.environ.get("AIGC_POLICY_PROFILE", "operational_low_false_ai")
if DEFAULT_POLICY_ID not in POLICY_BY_ID:
    DEFAULT_POLICY_ID = str(POLICY_PROFILES[0]["profile_id"])

if "AIGC_GENERATED_THRESHOLD" in os.environ:
    POLICY_BY_ID[DEFAULT_POLICY_ID]["generated_threshold"] = float(os.environ["AIGC_GENERATED_THRESHOLD"])
if "AIGC_REAL_THRESHOLD" in os.environ:
    POLICY_BY_ID[DEFAULT_POLICY_ID]["real_threshold"] = float(os.environ["AIGC_REAL_THRESHOLD"])
GENERATED_THRESHOLD = float(POLICY_BY_ID[DEFAULT_POLICY_ID]["generated_threshold"])
REAL_THRESHOLD = float(POLICY_BY_ID[DEFAULT_POLICY_ID]["real_threshold"])


def select_policy_profile(profile_id: str | None) -> dict[str, float | str]:
    if profile_id and profile_id in POLICY_BY_ID:
        return POLICY_BY_ID[profile_id]
    return POLICY_BY_ID[DEFAULT_POLICY_ID]


def read_top_features(path: Path, top_n: int = 8) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:top_n]


FEATURE_LABEL_MAP = {
    "is_png": "PNG 导出格式",
    "is_jpeg": "JPEG 导出格式",
    "is_webp": "WEBP 导出格式",
    "exif_present": "EXIF 是否存在",
    "info_key_count": "信息键数量",
    "band_count": "颜色通道数量",
    "has_alpha": "是否带 alpha 通道",
    "orig_width": "原始宽度",
    "orig_height": "原始高度",
    "aspect_ratio": "长宽比",
    "megapixels": "像素规模",
    "file_size_kb": "文件大小",
    "file_size_log": "文件大小对数",
}

FEATURE_EXPLANATION_MAP = {
    "is_png": "对应平台是否偏好以 PNG 导出图片。",
    "is_jpeg": "对应平台是否偏好以 JPEG 导出图片。",
    "is_webp": "对应平台是否偏好以 WEBP 导出图片。",
    "exif_present": "对应平台是否在导出图像中保留 EXIF 信息。",
    "info_key_count": "对应平台是否在文件信息区留下额外字段。",
    "band_count": "对应 RGB / RGBA 等通道结构差异。",
    "has_alpha": "对应透明通道是否被保留。",
    "orig_width": "对应平台输出宽度规格。",
    "orig_height": "对应平台输出高度规格。",
    "aspect_ratio": "对应平台常见输出画幅结构。",
    "megapixels": "对应平台输出的整体像素规模。",
    "file_size_kb": "对应平台压缩与编码策略差异。",
    "file_size_log": "对应文件体量分布差异。",
}


def normalize_feature_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        key = row.get("feature", "")
        normalized.append(
            {
                "feature": FEATURE_LABEL_MAP.get(key, key),
                "importance": f"{float(row.get('importance', 0.0)):.3f}",
                "explanation": FEATURE_EXPLANATION_MAP.get(key, "对应当前模型可利用的一类稳定差异。"),
            }
        )
    return normalized


GLOBAL_BINARY_FEATURES = normalize_feature_rows(read_top_features(BINARY_TOP))
GLOBAL_PLATFORM_FEATURES = normalize_feature_rows(read_top_features(PLATFORM_TOP))


def platform_label_text(platform_id: str) -> str:
    mapping = {
        "PLT01": "文心一言（PLT01）",
        "PLT02": "通义千问（PLT02）",
        "PLT03": "即梦AI / 字节系文生图（PLT03）",
        "PLT05": "智谱GLM-Image / 清言相关文生图（PLT05）",
        "real": "真实图片",
        "uncertain": "需人工复核",
    }
    return mapping.get(platform_id, platform_id)


def format_signal_snapshot(snapshot: dict[str, float]) -> list[dict[str, str]]:
    return [
        {"label": "原始尺寸", "value": f"{int(snapshot['orig_width'])} x {int(snapshot['orig_height'])}"},
        {"label": "纵横比", "value": f"{snapshot['aspect_ratio']:.3f}"},
        {"label": "像素规模", "value": f"{snapshot['megapixels']:.3f} MP"},
        {"label": "文件大小", "value": f"{snapshot['file_size_kb']:.1f} KB"},
        {"label": "PNG", "value": "是" if snapshot["is_png"] >= 0.5 else "否"},
        {"label": "JPEG", "value": "是" if snapshot["is_jpeg"] >= 0.5 else "否"},
        {"label": "WEBP", "value": "是" if snapshot["is_webp"] >= 0.5 else "否"},
        {"label": "Alpha通道", "value": "是" if snapshot["has_alpha"] >= 0.5 else "否"},
        {"label": "EXIF", "value": "有" if snapshot["exif_present"] >= 0.5 else "无"},
        {"label": "信息键数量", "value": str(int(snapshot["info_key_count"]))},
        {"label": "通道数", "value": str(int(snapshot["band_count"]))},
    ]


def build_rationale(prediction: dict) -> list[str]:
    lines: list[str] = []
    snap = prediction["feature_snapshot"]
    profile = prediction.get("policy_profile", {})
    if profile:
        lines.append(
            f"当前使用{profile.get('profile_name', '默认')}策略：真实阈值 {prediction['real_threshold']:.2f}，AI 阈值 {prediction['generated_threshold']:.2f}。{profile.get('ui_note', '')}"
        )
    if prediction["pred_binary_label"] == "real":
        lines.append(
            f"当前模型先做 AI/real 二分类，AI 生成概率为 {prediction['pred_prob_generated']:.4f}，低于真实图阈值 {prediction['real_threshold']:.2f}，因此判为 real，不继续给出平台来源。"
        )
    elif prediction["pred_binary_label"] == "uncertain":
        lines.append(
            f"当前 AI 生成概率为 {prediction['pred_prob_generated']:.4f}，落在 {prediction['real_threshold']:.2f} 到 {prediction['generated_threshold']:.2f} 的复核区间内。系统不会把这类图片硬判为 AI，也不会继续做平台归因。"
        )
        lines.append("证件照、白底商品图、截图和压缩后的真实图容易呈现背景干净、留痕较少、尺寸规整等特征，当前版本将这类边界样本优先交给人工复核。")
    else:
        lines.append(
            f"当前模型先判断为 generated，概率为 {prediction['pred_prob_generated']:.4f}，达到 AI 阈值 {prediction['generated_threshold']:.2f}；随后在已采样平台中做归因，当前给出的平台是 {platform_label_text(prediction['pred_final_label'])}。"
        )
    lines.append("系统当前主要依赖文件层与导出链路信号，并结合增强视觉统计特征，而不是只看图片语义内容。")

    signal_parts = []
    if snap["is_png"] >= 0.5:
        signal_parts.append("PNG 导出")
    if snap["is_jpeg"] >= 0.5:
        signal_parts.append("JPEG 导出")
    if snap["has_alpha"] >= 0.5:
        signal_parts.append("带 alpha 通道")
    if snap["exif_present"] >= 0.5:
        signal_parts.append("存在 EXIF")
    if snap["info_key_count"] > 0:
        signal_parts.append(f"info 字段数 {int(snap['info_key_count'])}")
    signal_parts.append(f"尺寸 {int(snap['orig_width'])}x{int(snap['orig_height'])}")
    lines.append("这张图当前被模型重点利用的直接信号包括：" + "、".join(signal_parts) + "。")
    if prediction["pred_binary_label"] == "generated":
        lines.append("因此，这个平台归因结果应理解为当前采样平台导出特征下的归因原型，不应表述成开放世界稳定来源鉴定。")
    else:
        lines.append("因此，本次结果应理解为低成本初筛；真实证件照、证书照和白底商品图等边界样本仍应保留人工复核。")
    return lines


@app.get("/")
def index():
    return render_template(
        "index.html",
        global_binary_features=GLOBAL_BINARY_FEATURES[:4],
        global_binary_features_more=GLOBAL_BINARY_FEATURES[4:],
        global_platform_features=GLOBAL_PLATFORM_FEATURES[:4],
        global_platform_features_more=GLOBAL_PLATFORM_FEATURES[4:],
        policy_profiles=POLICY_PROFILES,
        default_policy_id=DEFAULT_POLICY_ID,
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "binary_model": str(BINARY_BUNDLE.relative_to(ROOT)),
            "platform_model": str(PLATFORM_BUNDLE.relative_to(ROOT)),
            "generated_threshold": GENERATED_THRESHOLD,
            "real_threshold": REAL_THRESHOLD,
            "default_policy_profile": DEFAULT_POLICY_ID,
            "policy_profiles": POLICY_PROFILES,
        }
    )


@app.post("/api/predict")
def predict_api():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "missing file"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"status": "error", "message": "empty filename"}), 400

    raw = file.read()
    if not raw:
        return jsonify({"status": "error", "message": "empty file"}), 400

    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        return jsonify({"status": "error", "message": "unsupported or broken image"}), 400

    suffix = Path(file.filename).suffix or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            temp_path = Path(tmp.name)
        policy = select_policy_profile(request.form.get("policy_profile"))
        prediction = predict_one(
            temp_path,
            binary_bundle,
            platform_bundle,
            generated_threshold=float(policy["generated_threshold"]),
            real_threshold=float(policy["real_threshold"]),
        )
        prediction["policy_profile"] = policy
    finally:
        if temp_path and temp_path.exists():
            os.unlink(temp_path)

    signal_snapshot = format_signal_snapshot(prediction["feature_snapshot"])
    rationale = build_rationale(prediction)

    return jsonify(
        {
            "status": "ok",
            "result": {
                "binary_label": prediction["pred_binary_label"],
                "decision_status": prediction["decision_status"],
                "decision_text": prediction["decision_text"],
                "generated_probability": prediction["pred_prob_generated"],
                "platform_label": prediction["pred_final_label"],
                "platform_label_text": platform_label_text(prediction["pred_final_label"])
                if prediction["pred_binary_label"] == "generated"
                else prediction["decision_text"],
                "platform_probabilities": prediction["pred_platform_probabilities"]
                if prediction["pred_binary_label"] == "generated"
                else {},
                "signal_snapshot": signal_snapshot,
                "rationale": rationale,
                "thresholds": {
                    "real": prediction["real_threshold"],
                    "generated": prediction["generated_threshold"],
                },
                "policy_profile_id": prediction["policy_profile"]["profile_id"],
                "policy_profile_name": prediction["policy_profile"]["profile_name"],
                "policy_profile_note": prediction["policy_profile"]["ui_note"],
                "policy_metrics": {
                    "test_real_false_ai_rate": prediction["policy_profile"].get("test_real_false_ai_rate"),
                    "test_generated_auto_recall": prediction["policy_profile"].get("test_generated_auto_recall"),
                    "test_generated_risk_capture_rate": prediction["policy_profile"].get("test_generated_risk_capture_rate"),
                    "test_review_rate": prediction["policy_profile"].get("test_review_rate"),
                },
            },
            "global_binary_features": GLOBAL_BINARY_FEATURES,
            "global_platform_features": GLOBAL_PLATFORM_FEATURES,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    app.run(host="0.0.0.0", port=port, debug=False)
