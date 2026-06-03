from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


FEATURE_MODE_BASIC = "basic"
FEATURE_MODE_ENHANCED = "enhanced"
FEATURE_MODE_ENHANCED_META = "enhanced_meta"
VALID_FEATURE_MODES = {FEATURE_MODE_BASIC, FEATURE_MODE_ENHANCED, FEATURE_MODE_ENHANCED_META}


def _load_rgb_array(image_path: Path, size: int = 128) -> tuple[Image.Image, np.ndarray]:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB").resize((size, size))
        arr = np.asarray(rgb).astype("float32") / 255.0
    return rgb, arr


def _base_features(rgb: Image.Image, arr: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {
        "width": float(rgb.width),
        "height": float(rgb.height),
        "mean_r": float(arr[:, :, 0].mean()),
        "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
        "std_r": float(arr[:, :, 0].std()),
        "std_g": float(arr[:, :, 1].std()),
        "std_b": float(arr[:, :, 2].std()),
        "brightness_mean": float(arr.mean()),
    }

    bins = np.linspace(0.0, 1.0, 9)
    for channel_idx, channel_name in enumerate(["r", "g", "b"]):
        hist, _ = np.histogram(arr[:, :, channel_idx], bins=bins)
        hist = hist.astype("float32") / max(float(hist.sum()), 1.0)
        for i, value in enumerate(hist):
            feats[f"hist_{channel_name}_{i}"] = float(value)

    return feats


def _enhanced_features(arr: np.ndarray) -> dict[str, float]:
    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    dxx = np.diff(dx, axis=1)
    dyy = np.diff(dy, axis=0)
    lap_var = float(dxx.var() + dyy.var())

    cmax = arr.max(axis=2)
    cmin = arr.min(axis=2)
    saturation = np.where(cmax > 1e-6, (cmax - cmin) / np.maximum(cmax, 1e-6), 0.0)

    corr_rg = float(np.corrcoef(arr[:, :, 0].ravel(), arr[:, :, 1].ravel())[0, 1])
    corr_rb = float(np.corrcoef(arr[:, :, 0].ravel(), arr[:, :, 2].ravel())[0, 1])
    corr_gb = float(np.corrcoef(arr[:, :, 1].ravel(), arr[:, :, 2].ravel())[0, 1])

    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = float(radius.max())
    band_edges = np.linspace(0.0, max_r, 6)

    band_feats: dict[str, float] = {}
    power_total = float(power.sum()) if float(power.sum()) > 0 else 1.0
    for i in range(5):
        mask = (radius >= band_edges[i]) & (radius < band_edges[i + 1])
        band_power = float(power[mask].sum()) / power_total
        band_feats[f"fft_band_{i}"] = band_power
    high_freq_ratio = band_feats["fft_band_3"] + band_feats["fft_band_4"]

    feats = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p10": float(np.percentile(gray, 10)),
        "gray_p90": float(np.percentile(gray, 90)),
        "edge_mean": float(grad_mag.mean()),
        "edge_std": float(grad_mag.std()),
        "laplacian_var": lap_var,
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "channel_corr_rg": corr_rg,
        "channel_corr_rb": corr_rb,
        "channel_corr_gb": corr_gb,
        "high_freq_ratio": float(high_freq_ratio),
    }
    feats.update(band_feats)
    return feats


def _metadata_features(image_path: Path) -> dict[str, float]:
    file_size = float(os.path.getsize(image_path)) if image_path.exists() else 0.0
    with Image.open(image_path) as im:
        width, height = im.size
        fmt = (im.format or "").upper()
        bands = im.getbands()
        info = getattr(im, "info", {}) or {}
        try:
            exif_obj = im.getexif()
            exif_present = float(bool(exif_obj and len(exif_obj) > 0))
        except Exception:
            exif_present = 0.0

    aspect_ratio = float(width / max(height, 1))
    megapixels = float(width * height) / 1_000_000.0
    has_alpha = float("A" in bands)
    info_key_count = float(len(info))

    return {
        "orig_width": float(width),
        "orig_height": float(height),
        "aspect_ratio": aspect_ratio,
        "megapixels": megapixels,
        "file_size_kb": file_size / 1024.0,
        "file_size_log": float(np.log1p(file_size)),
        "has_alpha": has_alpha,
        "band_count": float(len(bands)),
        "is_png": float(fmt == "PNG"),
        "is_jpeg": float(fmt in {"JPEG", "JPG"}),
        "is_webp": float(fmt == "WEBP"),
        "exif_present": exif_present,
        "info_key_count": info_key_count,
    }


def compute_basic_image_features(image_path: Path) -> dict[str, float]:
    rgb, arr = _load_rgb_array(image_path)
    return _base_features(rgb, arr)


def compute_enhanced_image_features(image_path: Path) -> dict[str, float]:
    rgb, arr = _load_rgb_array(image_path)
    feats = _base_features(rgb, arr)
    feats.update(_enhanced_features(arr))
    return feats


def compute_enhanced_meta_image_features(image_path: Path) -> dict[str, float]:
    rgb, arr = _load_rgb_array(image_path)
    feats = _base_features(rgb, arr)
    feats.update(_enhanced_features(arr))
    feats.update(_metadata_features(image_path))
    return feats


def compute_image_features(image_path: Path, mode: str = FEATURE_MODE_BASIC) -> dict[str, float]:
    if mode not in VALID_FEATURE_MODES:
        raise ValueError(f"unsupported feature mode: {mode}")
    if mode == FEATURE_MODE_BASIC:
        return compute_basic_image_features(image_path)
    if mode == FEATURE_MODE_ENHANCED:
        return compute_enhanced_image_features(image_path)
    return compute_enhanced_meta_image_features(image_path)
