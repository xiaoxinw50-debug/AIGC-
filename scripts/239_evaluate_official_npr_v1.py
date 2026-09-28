#!/usr/bin/env python3
"""Evaluate official NPR checkpoints without tuning on any test split."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/NPR-DeepfakeDetection-main"
MODERN_MANIFEST = ROOT / "data/processed/experiment_tables/modern_ai_detection_model_subset_v1.csv"
AIGEN_MANIFEST = ROOT / "data/processed/experiment_tables/aigenimages2026_manifest_v1.csv"
OUT_DIR = ROOT / "outputs/reports/official_npr_v1"
TABLE_DIR = ROOT / "outputs/tables"
VALID_MANIFEST_OUT = ROOT / "data/processed/experiment_tables/npr_validation_manifest_v1.csv"
TEST_MANIFEST_OUT = ROOT / "data/processed/experiment_tables/npr_test_manifest_v1.csv"
RANDOM_STATE = 42
ImageFile.LOAD_TRUNCATED_IMAGES = True

CANDIDATES = {
    "official_npr_checkpoint": UPSTREAM / "NPR.pth",
    "official_npr_3090_state_dict": UPSTREAM / "model_epoch_last_3090.pth",
}

VALID_SCOPE_ORDER = [
    "community_validation",
    "legacy_validation",
    "ntire_standard_adapt_validation",
    "aigen2026_adapt_validation",
]
TEST_SCOPE_ORDER = [
    "community_generator_holdout_test",
    "legacy_comparability_test",
    "native_export_test",
    "ntire_standard_adapt_test",
    "ntire_hard_external_test",
    "qwen_unseen_generator_test",
    "safeimg_external_test",
    "aigen2026_official_external_test",
    "aigen2026_jpeg_q90_stress_test",
    "aigen2026_resize75_jpeg85_stress_test",
]


def stable_score(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def assign_ntire_protocol(frame: pd.DataFrame) -> pd.Series:
    assignment = pd.Series("not_ntire_standard", index=frame.index, dtype="object")
    standard = frame[
        frame["source_family"].eq("ntire2026_validation") & frame["category"].eq("standard")
    ]
    for _, part in standard.groupby("label"):
        ordered = part.assign(_score=part["experiment_sample_id"].astype(str).map(stable_score)).sort_values(
            ["_score", "experiment_sample_id"]
        )
        train_count = int(round(len(ordered) * 0.20))
        valid_count = int(round(len(ordered) * 0.10))
        assignment.loc[ordered.index[:train_count]] = "adapt_train"
        assignment.loc[ordered.index[train_count : train_count + valid_count]] = "adapt_valid"
        assignment.loc[ordered.index[train_count + valid_count :]] = "adapt_test"
    return assignment


def assign_aigen_protocol(frame: pd.DataFrame) -> pd.Series:
    assignment = pd.Series("official_external_test", index=frame.index, dtype="object")
    train = frame[frame["protocol_split"].eq("train_pool")]
    for _, part in train.groupby("label"):
        ordered = part.assign(_score=part["experiment_sample_id"].astype(str).map(stable_score)).sort_values(
            ["_score", "experiment_sample_id"]
        )
        valid_count = int(round(len(ordered) * 0.10))
        assignment.loc[ordered.index[:valid_count]] = "adapt_valid"
        assignment.loc[ordered.index[valid_count:]] = "adapt_train"
    return assignment


def normalize_rows(frame: pd.DataFrame, scope: str, condition: str = "original") -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "experiment_sample_id": frame["experiment_sample_id"].astype(str),
            "image_path": frame["image_path"].astype(str),
            "label": frame["label"].astype(int),
            "label_name": frame["label_name"].astype(str),
            "source_family": frame["source_family"].astype(str),
            "generator": frame["generator"].fillna("").astype(str),
            "category": frame["category"].fillna("").astype(str),
            "eval_scope": scope,
            "condition": condition,
        }
    )
    if condition != "original":
        result["experiment_sample_id"] = result["experiment_sample_id"] + f"__{condition}"
    return result


def build_protocol_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    modern = pd.read_csv(MODERN_MANIFEST, low_memory=False)
    modern["ntire_split"] = assign_ntire_protocol(modern)
    aigen = pd.read_csv(AIGEN_MANIFEST, low_memory=False)
    aigen["aigen_split"] = assign_aigen_protocol(aigen)

    valid_parts = [
        normalize_rows(
            modern[
                modern["source_family"].eq("community_forensics_small")
                & modern["protocol_split"].eq("valid")
            ],
            "community_validation",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("legacy_cifake_genimage")
                & modern["protocol_split"].eq("valid")
            ],
            "legacy_validation",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("ntire2026_validation")
                & modern["ntire_split"].eq("adapt_valid")
            ],
            "ntire_standard_adapt_validation",
        ),
        normalize_rows(aigen[aigen["aigen_split"].eq("adapt_valid")], "aigen2026_adapt_validation"),
    ]
    test_parts = [
        normalize_rows(
            modern[
                modern["source_family"].eq("community_forensics_small")
                & modern["protocol_split"].eq("test")
            ],
            "community_generator_holdout_test",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("legacy_cifake_genimage")
                & modern["protocol_split"].eq("test")
            ],
            "legacy_comparability_test",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("native_platform_mixed")
                & modern["protocol_split"].eq("test")
            ],
            "native_export_test",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("ntire2026_validation")
                & modern["ntire_split"].eq("adapt_test")
            ],
            "ntire_standard_adapt_test",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("ntire2026_validation")
                & modern["category"].eq("hard")
            ],
            "ntire_hard_external_test",
        ),
        normalize_rows(
            modern[
                modern["source_family"].eq("qwen_image_bench") & modern["protocol_split"].eq("test")
            ],
            "qwen_unseen_generator_test",
        ),
        normalize_rows(modern[modern["source_family"].eq("safeimg")], "safeimg_external_test"),
    ]
    official = aigen[aigen["aigen_split"].eq("official_external_test")]
    test_parts.extend(
        [
            normalize_rows(official, "aigen2026_official_external_test"),
            normalize_rows(official, "aigen2026_jpeg_q90_stress_test", "jpeg_q90"),
            normalize_rows(
                official,
                "aigen2026_resize75_jpeg85_stress_test",
                "resize75_jpeg85",
            ),
        ]
    )
    valid = pd.concat(valid_parts, ignore_index=True).sort_values(
        ["eval_scope", "experiment_sample_id"]
    ).reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True).sort_values(
        ["eval_scope", "experiment_sample_id"]
    ).reset_index(drop=True)
    for name, frame in [("valid", valid), ("test", test)]:
        if frame["experiment_sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate sample ids in {name} manifest")
        missing = [path for path in frame["image_path"] if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing paths in {name} manifest")
    return valid, test


def apply_condition(image: Image.Image, condition: str) -> Image.Image:
    if condition == "original":
        return image
    if condition == "jpeg_q90":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, subsampling=2)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        result.load()
        buffer.close()
        return result
    if condition == "resize75_jpeg85":
        width, height = image.size
        reduced = image.resize(
            (max(2, int(round(width * 0.75))), max(2, int(round(height * 0.75)))),
            Image.Resampling.BILINEAR,
        )
        restored = reduced.resize((width, height), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        restored.save(buffer, format="JPEG", quality=85, subsampling=2)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        result.load()
        buffer.close()
        return result
    raise ValueError(f"Unknown condition: {condition}")


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(str(row["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image = apply_condition(image, str(row["condition"]))
        return self.transform(image), index


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(UPSTREAM))
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def run_inference(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    label: str,
) -> pd.DataFrame:
    dataset = ManifestDataset(frame)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits = np.full(len(frame), np.nan, dtype="float32")
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader):
            output = model(images.to(device)).reshape(-1).detach().cpu().numpy().astype("float32")
            logits[indices.numpy()] = output
            completed = min((batch_index + 1) * batch_size, len(frame))
            if completed % 1024 < batch_size or completed == len(frame):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"{label} progress={completed}/{len(frame)} rate={completed / elapsed:.1f} images/s",
                    flush=True,
                )
    if np.isnan(logits).any():
        raise RuntimeError(f"Incomplete inference for {label}")
    result = frame.copy()
    result["npr_logit"] = logits
    result["npr_probability"] = 1.0 / (1.0 + np.exp(-np.clip(logits.astype("float64"), -60.0, 60.0)))
    return result


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    result = 0.0
    for index in range(bins):
        mask = bucket == index
        if np.any(mask):
            result += float(mask.mean()) * abs(float(probability[mask].mean()) - float(y_true[mask].mean()))
    return float(result)


def metric_block(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    labels = set(y_true.tolist())
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "real_count": int((y_true == 0).sum()),
        "generated_count": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)) if len(labels) == 2 else None,
        "precision_generated": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "recall_generated": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "f1_generated": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(labels) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "expected_calibration_error_10bin": expected_calibration_error(y_true, probability),
        "mean_probability_generated": float(probability.mean()),
    }
    if len(labels) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        result.update(
            {
                "true_real_pred_real": int(tn),
                "true_real_pred_generated": int(fp),
                "true_generated_pred_real": int(fn),
                "true_generated_pred_generated": int(tp),
                "real_false_positive_rate": float(fp / max(tn + fp, 1)),
                "real_recall": float(tn / max(tn + fp, 1)),
            }
        )
    else:
        result.update(
            {
                "true_real_pred_real": None,
                "true_real_pred_generated": None,
                "true_generated_pred_real": int((prediction == 0).sum()) if 1 in labels else None,
                "true_generated_pred_generated": int((prediction == 1).sum()) if 1 in labels else None,
                "real_false_positive_rate": None,
                "real_recall": None,
            }
        )
    return result


def evaluate_scopes(frame: pd.DataFrame, threshold: float, policy_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, part in frame.groupby("eval_scope", sort=False):
        rows.append(
            {
                "policy_id": policy_id,
                "eval_scope": scope,
                **metric_block(
                    part["label"].to_numpy(dtype=int),
                    part["npr_probability"].to_numpy(dtype=float),
                    threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def select_candidate(candidate_metrics: pd.DataFrame) -> str:
    rows: list[tuple[float, str]] = []
    for candidate, part in candidate_metrics.groupby("candidate_id"):
        auc = part["roc_auc"].astype(float)
        score = float(auc.min() + auc.mean() - 0.15 * part["brier_score"].astype(float).max())
        rows.append((score, str(candidate)))
    return max(rows)[1]


def select_global_threshold(valid_predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    candidates = sorted(
        set(
            [0.5]
            + [float(1.0 / (1.0 + math.exp(-value))) for value in np.linspace(-14.0, 14.0, 281)]
        )
    )
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        metrics = evaluate_scopes(valid_predictions, threshold, "candidate")
        ba = metrics["balanced_accuracy"].astype(float)
        fpr = metrics["real_false_positive_rate"].astype(float)
        feasible = bool(float(fpr.max()) <= 0.20)
        score = float(ba.min() + ba.mean() - 0.50 * fpr.max()) if feasible else -1.0
        rows.append(
            {
                "threshold": threshold,
                "feasible_max_fpr_20pct": feasible,
                "min_balanced_accuracy": float(ba.min()),
                "mean_balanced_accuracy": float(ba.mean()),
                "max_real_fpr": float(fpr.max()),
                "selection_score": score,
            }
        )
    table = pd.DataFrame(rows)
    feasible = table[table["feasible_max_fpr_20pct"]]
    if feasible.empty:
        selected = table.sort_values(["max_real_fpr", "selection_score"], ascending=[True, False]).iloc[0]
    else:
        selected = feasible.sort_values(
            ["selection_score", "min_balanced_accuracy", "mean_balanced_accuracy"],
            ascending=False,
        ).iloc[0]
    return float(selected["threshold"]), table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas null values to strict JSON null values."""
    return json.loads(frame.to_json(orient="records"))


def selected_threshold_from_table(frame: pd.DataFrame) -> float:
    feasible = frame[frame["feasible_max_fpr_20pct"].astype(bool)]
    if feasible.empty:
        selected = frame.sort_values(["max_real_fpr", "selection_score"], ascending=[True, False]).iloc[0]
    else:
        selected = feasible.sort_values(
            ["selection_score", "min_balanced_accuracy", "mean_balanced_accuracy"],
            ascending=False,
        ).iloc[0]
    return float(selected["threshold"])


def write_summary(
    candidate_table: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    threshold_table: pd.DataFrame,
    validation_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    selected_id = select_candidate(candidate_table)
    selected_path = CANDIDATES[selected_id]
    selected_threshold = selected_threshold_from_table(threshold_table)
    summary = {
        "status": "completed",
        "upstream_repository": "https://github.com/chuangchuangtan/NPR-DeepfakeDetection",
        "upstream_commit": "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a",
        "selected_candidate": selected_id,
        "selected_weight_path": str(selected_path.relative_to(ROOT)),
        "selected_weight_sha256": sha256(selected_path),
        "official_threshold": 0.5,
        "validation_selected_global_threshold": selected_threshold,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation_scopes": VALID_SCOPE_ORDER,
        "test_scopes": TEST_SCOPE_ORDER,
        "candidate_validation_metrics": json_records(candidate_table),
        "validation_scope_metrics": json_records(valid_metrics),
        "test_scope_metrics": json_records(test_metrics),
        "license_boundary": (
            "The upstream snapshot has no explicit license file. NPR code and weights are research-only "
            "and are not eligible for public web deployment without separate license confirmation."
        ),
        "decision_boundary": (
            "The official zero-shot model is evaluated before any project-specific fine-tuning. "
            "Only validation scopes select the checkpoint and one global threshold; all test scopes remain untouched."
        ),
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (OUT_DIR / "summary.json").write_text(payload, encoding="utf-8")
    print(payload, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore cached predictions and rerun inference.")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cache_files = [
        VALID_MANIFEST_OUT,
        TEST_MANIFEST_OUT,
        TABLE_DIR / "official_npr_candidate_validation_metrics_v1.csv",
        TABLE_DIR / "official_npr_threshold_candidates_v1.csv",
        TABLE_DIR / "official_npr_validation_scope_metrics_v1.csv",
        TABLE_DIR / "official_npr_test_scope_metrics_v1.csv",
        OUT_DIR / "validation_predictions.csv",
        OUT_DIR / "test_predictions.csv",
    ]
    if not args.force and all(path.exists() for path in cache_files):
        print("Reusing completed NPR predictions and rebuilding strict JSON summary.", flush=True)
        write_summary(
            pd.read_csv(cache_files[2]),
            pd.read_csv(cache_files[4]),
            pd.read_csv(cache_files[5]),
            pd.read_csv(cache_files[3]),
            len(pd.read_csv(VALID_MANIFEST_OUT, low_memory=False)),
            len(pd.read_csv(TEST_MANIFEST_OUT, low_memory=False)),
        )
        return
    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    valid_manifest, test_manifest = build_protocol_manifests()
    valid_manifest.to_csv(VALID_MANIFEST_OUT, index=False, encoding="utf-8-sig")
    test_manifest.to_csv(TEST_MANIFEST_OUT, index=False, encoding="utf-8-sig")

    candidate_metrics: list[pd.DataFrame] = []
    validation_predictions: dict[str, pd.DataFrame] = {}
    for candidate_id, path in CANDIDATES.items():
        print(f"loading candidate={candidate_id} sha256={sha256(path)}", flush=True)
        model = load_model(path, device)
        predictions = run_inference(valid_manifest, model, device, batch_size, f"valid:{candidate_id}")
        validation_predictions[candidate_id] = predictions
        metrics = evaluate_scopes(predictions, 0.5, "official_threshold_0.5")
        metrics.insert(0, "candidate_id", candidate_id)
        candidate_metrics.append(metrics)
        del model

    candidate_table = pd.concat(candidate_metrics, ignore_index=True)
    selected_id = select_candidate(candidate_table)
    selected_path = CANDIDATES[selected_id]
    selected_valid = validation_predictions[selected_id]
    selected_threshold, threshold_table = select_global_threshold(selected_valid)
    valid_fixed = evaluate_scopes(selected_valid, 0.5, "official_threshold_0.5")
    valid_calibrated = evaluate_scopes(selected_valid, selected_threshold, "validation_selected_global_threshold")
    valid_metrics = pd.concat([valid_fixed, valid_calibrated], ignore_index=True)

    print(
        f"selected_candidate={selected_id} selected_threshold={selected_threshold:.8f} "
        f"valid_rows={len(valid_manifest)} test_rows={len(test_manifest)}",
        flush=True,
    )
    selected_model = load_model(selected_path, device)
    test_predictions = run_inference(test_manifest, selected_model, device, batch_size, f"test:{selected_id}")
    test_fixed = evaluate_scopes(test_predictions, 0.5, "official_threshold_0.5")
    test_calibrated = evaluate_scopes(
        test_predictions,
        selected_threshold,
        "validation_selected_global_threshold",
    )
    test_metrics = pd.concat([test_fixed, test_calibrated], ignore_index=True)

    aigen_original = test_predictions[
        test_predictions["eval_scope"].eq("aigen2026_official_external_test")
    ].copy()
    per_generator_rows: list[dict[str, Any]] = []
    for generator, part in aigen_original[aigen_original["label"].eq(1)].groupby("generator"):
        probability = part["npr_probability"].to_numpy(dtype=float)
        per_generator_rows.append(
            {
                "generator": generator,
                "n": len(part),
                "recall_at_official_threshold": float((probability >= 0.5).mean()),
                "recall_at_selected_threshold": float((probability >= selected_threshold).mean()),
                "mean_probability_generated": float(probability.mean()),
            }
        )

    candidate_table.to_csv(TABLE_DIR / "official_npr_candidate_validation_metrics_v1.csv", index=False)
    threshold_table.to_csv(TABLE_DIR / "official_npr_threshold_candidates_v1.csv", index=False)
    valid_metrics.to_csv(TABLE_DIR / "official_npr_validation_scope_metrics_v1.csv", index=False)
    test_metrics.to_csv(TABLE_DIR / "official_npr_test_scope_metrics_v1.csv", index=False)
    pd.DataFrame(per_generator_rows).sort_values("recall_at_selected_threshold").to_csv(
        TABLE_DIR / "official_npr_aigen_per_generator_metrics_v1.csv",
        index=False,
    )
    selected_valid.assign(candidate_id=selected_id).to_csv(
        OUT_DIR / "validation_predictions.csv",
        index=False,
    )
    test_predictions.assign(candidate_id=selected_id).to_csv(
        OUT_DIR / "test_predictions.csv",
        index=False,
    )

    write_summary(
        candidate_table,
        valid_metrics,
        test_metrics,
        threshold_table,
        len(valid_manifest),
        len(test_manifest),
    )


if __name__ == "__main__":
    main()
