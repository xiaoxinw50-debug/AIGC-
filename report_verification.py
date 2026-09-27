from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import ExifTags, Image


REPORT_SCHEMA_VERSION = "aigc-detection-report/1.0"
REPORT_SIGNATURE_ALGORITHM = "HMAC-SHA256"
MAX_REPORT_TOKEN_LENGTH = 12000
MAX_RECORDED_METADATA_KEYS = 64

FORMAL_LABEL_PATTERNS = (
    "aigc",
    "aigenerated",
    "syntheticcontent",
    "contentcredential",
    "c2pa",
)
GENERATION_CLUE_PATTERNS = (
    "prompt",
    "parameter",
    "workflow",
    "software",
    "usercomment",
    "imagedescription",
)


class ReportSigningUnavailable(RuntimeError):
    pass


class InvalidReportToken(ValueError):
    pass


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except Exception as exc:
        raise InvalidReportToken("报告凭证编码无效") from exc


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signing_key(secret: str | None = None) -> bytes:
    value = secret if secret is not None else os.environ.get("AIGC_REPORT_SIGNING_KEY", "")
    if len(value.encode("utf-8")) < 32:
        raise ReportSigningUnavailable("AIGC_REPORT_SIGNING_KEY 必须至少包含 32 个字节")
    return value.encode("utf-8")


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _matching_keys(keys: list[str], patterns: tuple[str, ...]) -> list[str]:
    matched = []
    for key in keys:
        normalized = _normalized_key(key)
        if any(pattern in normalized for pattern in patterns):
            matched.append(key)
    return sorted(set(matched))


def inspect_label_evidence(raw: bytes) -> dict[str, Any]:
    """Inspect readable metadata keys without retaining metadata values or the image."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            all_info_keys = sorted(str(key)[:120] for key in image.info.keys())
            exif = image.getexif()
            all_exif_keys = sorted(str(ExifTags.TAGS.get(tag_id, tag_id))[:120] for tag_id in exif.keys())
            image_format = str(image.format or "UNKNOWN").upper()
    except Exception:
        return {
            "explicit_label": {
                "status": "manual_review_required",
                "text": "显式标识需要结合图面人工核验。",
            },
            "implicit_label": {
                "status": "unreadable",
                "detected_keys": [],
                "text": "当前文件元数据无法读取。",
            },
            "generation_metadata_clues": {
                "status": "unreadable",
                "detected_keys": [],
            },
            "file_metadata": {
                "format": "UNKNOWN",
                "info_key_count": 0,
                "exif_key_count": 0,
                "info_keys": [],
                "exif_keys": [],
            },
            "standard_scope_note": "本系统仅检查当前解析器可读取的文件层字段，不构成标识合规认证。",
        }

    info_keys = all_info_keys[:MAX_RECORDED_METADATA_KEYS]
    exif_keys = all_exif_keys[:MAX_RECORDED_METADATA_KEYS]
    all_keys = all_info_keys + all_exif_keys
    formal_keys = _matching_keys(all_keys, FORMAL_LABEL_PATTERNS)[:MAX_RECORDED_METADATA_KEYS]
    clue_keys = [key for key in _matching_keys(all_keys, GENERATION_CLUE_PATTERNS) if key not in formal_keys][
        :MAX_RECORDED_METADATA_KEYS
    ]
    implicit_status = "detected" if formal_keys else "not_detected"
    return {
        "explicit_label": {
            "status": "manual_review_required",
            "text": "显式标识需要结合图面人工核验，当前模型不把视觉内容识别等同于显式标识认证。",
        },
        "implicit_label": {
            "status": implicit_status,
            "detected_keys": formal_keys,
            "text": "发现疑似 AIGC 隐式标识字段。" if formal_keys else "未在当前可读取元数据键中发现明确 AIGC 标识字段。",
        },
        "generation_metadata_clues": {
            "status": "detected" if clue_keys else "not_detected",
            "detected_keys": clue_keys,
        },
        "file_metadata": {
            "format": image_format,
            "info_key_count": len(all_info_keys),
            "exif_key_count": len(all_exif_keys),
            "info_keys": info_keys,
            "exif_keys": exif_keys,
        },
        "standard_scope_note": "本系统按照显式标识、隐式标识和模型推断分层记录证据；自动检查仅覆盖当前解析器可读取字段，不构成 GB 45438-2025 合规认证。",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_manifest(model_paths: dict[str, Path], open_set_version: str) -> dict[str, Any]:
    models = {}
    for name, path in sorted(model_paths.items()):
        if path.exists():
            models[name] = {
                "file": path.name,
                "sha256": sha256_file(path),
            }
    return {
        "open_set_version": open_set_version,
        "models": models,
    }


def _safe_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number > 1:
        return None
    return round(number, 8)


def build_report_payload(
    *,
    file_name: str,
    file_sha256: str,
    result: dict[str, Any],
    label_evidence: dict[str, Any],
    model_manifest: dict[str, Any],
    issuer: str | None = None,
) -> dict[str, Any]:
    issued_at = datetime.now(timezone.utc).replace(microsecond=0)
    report_id = f"AIGC-{issued_at:%Y%m%d}-{file_sha256[:8].upper()}-{secrets.token_hex(3).upper()}"
    source_thresholds = result.get("platform_open_set_thresholds") or {}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "issuer": issuer or os.environ.get("AIGC_REPORT_ISSUER", "AIGC 标识治理研究项目"),
        "file": {
            "name": Path(file_name).name[:255],
            "sha256": file_sha256,
            "format": label_evidence.get("file_metadata", {}).get("format", "UNKNOWN"),
        },
        "decision": {
            "binary_label": result.get("binary_label", ""),
            "generated_probability": _safe_probability(result.get("generated_probability")),
            "platform_label": result.get("platform_label", ""),
            "platform_accepted": result.get("platform_accepted") is True,
            "platform_confidence": _safe_probability(result.get("platform_confidence")),
            "knownness_score": _safe_probability(result.get("platform_knownness_score")),
            "risk_level": result.get("risk_level", ""),
        },
        "policy": {
            "profile_id": result.get("policy_profile_id", ""),
            "real_threshold": _safe_probability((result.get("thresholds") or {}).get("real")),
            "generated_threshold": _safe_probability((result.get("thresholds") or {}).get("generated")),
            "platform_confidence_threshold": _safe_probability(source_thresholds.get("confidence")),
            "platform_margin_threshold": _safe_probability(source_thresholds.get("margin")),
            "knownness_threshold": _safe_probability(source_thresholds.get("knownness")),
        },
        "label_evidence": label_evidence,
        "model_release": model_manifest,
        "limitations": [
            "报告验证仅证明签发内容未被修改，不证明上传图片在拍摄或传播过程中未被篡改。",
            "模型结果是风险判断与来源线索，不构成真实性证明、平台认证或单独定责依据。",
            "未发现隐式标识不等于图片不是 AI 生成，截图、压缩和转存可能移除文件层字段。",
        ],
    }


def issue_report(
    payload: dict[str, Any],
    verification_base_url: str,
    secret: str | None = None,
) -> dict[str, Any]:
    key = _signing_key(secret)
    canonical = _canonical_json(payload)
    encoded_payload = _base64url_encode(canonical)
    signature = hmac.new(key, canonical, hashlib.sha256).digest()
    token = f"{encoded_payload}.{_base64url_encode(signature)}"
    base_url = verification_base_url.rstrip("/")
    return {
        "status": "signed",
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": payload["report_id"],
        "issued_at": payload["issued_at"],
        "issuer": payload["issuer"],
        "algorithm": REPORT_SIGNATURE_ALGORITHM,
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "token": token,
        "verification_url": f"{base_url}/verify#token={quote(token, safe='')}",
    }


def verify_report_token(token: str, secret: str | None = None) -> dict[str, Any]:
    if not token or len(token) > MAX_REPORT_TOKEN_LENGTH:
        raise InvalidReportToken("报告凭证为空或长度异常")
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
    except ValueError as exc:
        raise InvalidReportToken("报告凭证结构无效") from exc
    canonical = _base64url_decode(encoded_payload)
    supplied_signature = _base64url_decode(encoded_signature)
    expected_signature = hmac.new(_signing_key(secret), canonical, hashlib.sha256).digest()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise InvalidReportToken("报告签名不匹配，内容可能已被修改")
    try:
        payload = json.loads(canonical.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidReportToken("报告内容无法解析") from exc
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise InvalidReportToken("报告版本不受支持")
    return {
        "valid": True,
        "algorithm": REPORT_SIGNATURE_ALGORITHM,
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "payload": payload,
    }


def signing_ready(secret: str | None = None) -> bool:
    try:
        _signing_key(secret)
    except ReportSigningUnavailable:
        return False
    return True
