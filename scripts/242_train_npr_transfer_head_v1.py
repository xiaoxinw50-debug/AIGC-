#!/usr/bin/env python3
"""Train and evaluate a leakage-safe linear head on frozen official NPR embeddings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = ROOT / "outputs/features/npr_official_embeddings_v1"
TABLE_DIR = ROOT / "outputs/tables"
OUT_DIR = ROOT / "outputs/reports/npr_transfer_head_v1"
FUSION_HELPER_PATH = ROOT / "scripts/240_evaluate_npr_fusion_v1.py"
BASE_TEST_PATH = ROOT / "outputs/reports/three_domain_moe_v1/predictions.csv"
RANDOM_STATE = 42

VALID_SCOPES = [
    "community_validation",
    "legacy_validation",
    "ntire_standard_adapt_validation",
    "aigen2026_adapt_validation",
]
TWO_CLASS_TEST_SCOPES = [
    "community_generator_holdout_test",
    "legacy_comparability_test",
    "native_export_test",
    "ntire_standard_adapt_test",
    "ntire_hard_external_test",
    "aigen2026_official_external_test",
]
UNKNOWN_GENERATED_SCOPES = ["qwen_unseen_generator_test", "safeimg_external_test"]
WEIGHT_STRATEGIES = ["uniform", "class_balanced", "domain_class_balanced"]
C_GRID = [0.001, 0.01, 0.1, 1.0]
THRESHOLD_GRID = np.unique(np.append(np.linspace(0.05, 0.95, 181), 0.5))
FUSION_WEIGHT_GRID = np.linspace(0.0, 0.50, 21)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    manifest_path = TABLE_DIR / f"npr_official_embedding_{split}_manifest_v1.csv"
    embeddings_path = EMBEDDING_DIR / split / "embeddings.npy"
    status_path = EMBEDDING_DIR / split / "status.npy"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    status = np.load(status_path, mmap_mode="r")
    if embeddings.shape != (len(manifest), 512):
        raise RuntimeError(f"Unexpected {split} embedding shape: {embeddings.shape}")
    if status.shape != (len(manifest),) or not np.all(status == 1):
        raise RuntimeError(f"Incomplete {split} embeddings")
    if not np.array_equal(manifest["embedding_row"].to_numpy(dtype=int), np.arange(len(manifest))):
        raise RuntimeError(f"Non-contiguous {split} embedding rows")
    return manifest, embeddings


def sample_weights(frame: pd.DataFrame, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.ones(len(frame), dtype="float64")
    if strategy == "class_balanced":
        groups = frame["label"].astype(str)
    elif strategy == "domain_class_balanced":
        groups = frame["eval_scope"].astype(str) + "::" + frame["label"].astype(str)
    else:
        raise ValueError(strategy)
    counts = groups.value_counts()
    weights = groups.map(lambda value: len(frame) / (len(counts) * counts[value])).to_numpy(dtype="float64")
    return weights / weights.mean()


def metric_block(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    labels = set(y_true.tolist())
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "real_count": int((y_true == 0).sum()),
        "generated_count": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, prediction)) if len(labels) == 2 else None
        ),
        "precision_generated": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_generated": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_generated": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(labels) == 2 else None,
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


def evaluate_by_scope(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    method_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, indices in frame.groupby("eval_scope", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        rows.append(
            {
                "method_id": method_id,
                "eval_scope": scope,
                **metric_block(
                    frame.iloc[positions]["label"].to_numpy(dtype=int),
                    probability[positions],
                    threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def selection_summary(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = evaluate_by_scope(frame, probability, threshold, "candidate")
    metrics = metrics[metrics["eval_scope"].isin(VALID_SCOPES)]
    ba = metrics["balanced_accuracy"].astype(float)
    fpr = metrics["real_false_positive_rate"].astype(float)
    min_ba = float(ba.min())
    mean_ba = float(ba.mean())
    max_fpr = float(fpr.max())
    mean_fpr = float(fpr.mean())
    std_ba = float(ba.std(ddof=0))
    # Worst-domain performance is primary; a high single-domain real-image FPR is penalized.
    score = min_ba + 0.50 * mean_ba - 0.35 * max_fpr - 0.10 * std_ba
    return {
        "min_balanced_accuracy": min_ba,
        "mean_balanced_accuracy": mean_ba,
        "max_real_fpr": max_fpr,
        "mean_real_fpr": mean_fpr,
        "std_balanced_accuracy": std_ba,
        "selection_score": float(score),
    }


def candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(record["selection_score"]),
        float(record["min_balanced_accuracy"]),
        float(record["mean_balanced_accuracy"]),
        -float(record["max_real_fpr"]),
    )


def fit_head_candidates(
    train: pd.DataFrame,
    train_x: np.ndarray,
    valid: pd.DataFrame,
    valid_x: np.ndarray,
) -> tuple[Pipeline, dict[str, Any], np.ndarray, pd.DataFrame]:
    y_train = train["label"].to_numpy(dtype=int)
    candidate_rows: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_record: dict[str, Any] | None = None
    best_probability: np.ndarray | None = None
    for strategy in WEIGHT_STRATEGIES:
        weights = sample_weights(train, strategy)
        for c_value in C_GRID:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c_value,
                            max_iter=3000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            model.fit(train_x, y_train, classifier__sample_weight=weights)
            probability = model.predict_proba(valid_x)[:, 1]
            model_best: dict[str, Any] | None = None
            for threshold in THRESHOLD_GRID:
                summary = selection_summary(valid, probability, float(threshold))
                record = {
                    "weight_strategy": strategy,
                    "regularization_c": float(c_value),
                    "threshold": float(threshold),
                    **summary,
                }
                candidate_rows.append(record)
                if model_best is None or candidate_sort_key(record) > candidate_sort_key(model_best):
                    model_best = record
            if model_best is None:
                raise RuntimeError("No head threshold candidate was evaluated")
            print(
                "head_candidate "
                f"strategy={strategy} C={c_value:g} threshold={model_best['threshold']:.3f} "
                f"min_ba={model_best['min_balanced_accuracy']:.4f} "
                f"mean_ba={model_best['mean_balanced_accuracy']:.4f} "
                f"max_fpr={model_best['max_real_fpr']:.4f}",
                flush=True,
            )
            if best_record is None or candidate_sort_key(model_best) > candidate_sort_key(best_record):
                best_model = model
                best_record = model_best
                best_probability = probability
    if best_model is None or best_record is None or best_probability is None:
        raise RuntimeError("No NPR transfer head selected")
    return best_model, best_record, best_probability, pd.DataFrame(candidate_rows)


def search_fusion(
    valid: pd.DataFrame,
    base_probability: np.ndarray,
    npr_probability: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_probability: np.ndarray | None = None
    for weight in FUSION_WEIGHT_GRID:
        probability = (1.0 - weight) * base_probability + weight * npr_probability
        weight_best: dict[str, Any] | None = None
        for threshold in THRESHOLD_GRID:
            record = {
                "npr_head_weight": float(weight),
                "threshold": float(threshold),
                **selection_summary(valid, probability, float(threshold)),
            }
            rows.append(record)
            if weight_best is None or candidate_sort_key(record) > candidate_sort_key(weight_best):
                weight_best = record
        if weight_best is None:
            raise RuntimeError("No fusion threshold candidate was evaluated")
        if best_record is None or candidate_sort_key(weight_best) > candidate_sort_key(best_record):
            best_record = weight_best
            best_probability = probability
    if best_record is None or best_probability is None:
        raise RuntimeError("No fusion candidate selected")
    return best_record, best_probability, pd.DataFrame(rows)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    train, train_x_memmap = load_split("train")
    valid, valid_x_memmap = load_split("valid")
    test, test_x_memmap = load_split("test")
    stress, stress_x_memmap = load_split("stress")
    train_x = np.asarray(train_x_memmap, dtype="float32")
    valid_x = np.asarray(valid_x_memmap, dtype="float32")
    test_x = np.asarray(test_x_memmap, dtype="float32")
    stress_x = np.asarray(stress_x_memmap, dtype="float32")

    model, selected_head, valid_head_probability, head_candidates = fit_head_candidates(
        train, train_x, valid, valid_x
    )
    test_head_probability = model.predict_proba(test_x)[:, 1]
    stress_head_probability = model.predict_proba(stress_x)[:, 1]
    head_threshold = float(selected_head["threshold"])

    fusion_helper = load_module(FUSION_HELPER_PATH, "npr_fusion_helper_v1")
    valid_with_base = fusion_helper.add_three_domain_validation_probability(valid.copy())
    valid_base_probability = valid_with_base["probability_generated"].to_numpy(dtype=float)
    fusion_selected, valid_fusion_probability, fusion_candidates = search_fusion(
        valid, valid_base_probability, valid_head_probability
    )

    base_test = pd.read_csv(BASE_TEST_PATH, low_memory=False)
    base_lookup = base_test.set_index(["experiment_sample_id", "eval_scope"])["probability_generated"]
    test_keys = pd.MultiIndex.from_frame(test[["experiment_sample_id", "eval_scope"]])
    test_base_probability = base_lookup.reindex(test_keys).to_numpy(dtype=float)
    if np.isnan(test_base_probability).any():
        raise RuntimeError("Missing three-domain baseline test probabilities")
    fusion_weight = float(fusion_selected["npr_head_weight"])
    test_fusion_probability = (
        (1.0 - fusion_weight) * test_base_probability + fusion_weight * test_head_probability
    )
    fusion_threshold = float(fusion_selected["threshold"])

    validation_metrics = pd.concat(
        [
            evaluate_by_scope(valid, valid_base_probability, 0.5, "three_domain_baseline"),
            evaluate_by_scope(
                valid,
                valid_head_probability,
                head_threshold,
                "validation_selected_npr_transfer_head",
            ),
            evaluate_by_scope(
                valid,
                valid_fusion_probability,
                fusion_threshold,
                "validation_selected_baseline_npr_head_fusion",
            ),
        ],
        ignore_index=True,
    )
    test_metrics = pd.concat(
        [
            evaluate_by_scope(test, test_base_probability, 0.5, "three_domain_baseline"),
            evaluate_by_scope(
                test,
                test_head_probability,
                head_threshold,
                "validation_selected_npr_transfer_head",
            ),
            evaluate_by_scope(
                test,
                test_fusion_probability,
                fusion_threshold,
                "validation_selected_baseline_npr_head_fusion",
            ),
        ],
        ignore_index=True,
    )
    stress_metrics = evaluate_by_scope(
        stress,
        stress_head_probability,
        head_threshold,
        "validation_selected_npr_transfer_head",
    )

    validation_predictions = valid.copy()
    validation_predictions["three_domain_probability"] = valid_base_probability
    validation_predictions["npr_transfer_probability"] = valid_head_probability
    validation_predictions["fusion_probability"] = valid_fusion_probability
    test_predictions = test.copy()
    test_predictions["three_domain_probability"] = test_base_probability
    test_predictions["npr_transfer_probability"] = test_head_probability
    test_predictions["fusion_probability"] = test_fusion_probability
    stress_predictions = stress.copy()
    stress_predictions["npr_transfer_probability"] = stress_head_probability

    head_candidates.to_csv(TABLE_DIR / "npr_transfer_head_validation_candidates_v1.csv", index=False)
    fusion_candidates.to_csv(TABLE_DIR / "npr_transfer_fusion_validation_candidates_v1.csv", index=False)
    validation_metrics.to_csv(TABLE_DIR / "npr_transfer_validation_scope_metrics_v1.csv", index=False)
    test_metrics.to_csv(TABLE_DIR / "npr_transfer_test_scope_metrics_v1.csv", index=False)
    stress_metrics.to_csv(TABLE_DIR / "npr_transfer_stress_scope_metrics_v1.csv", index=False)
    validation_predictions.to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    test_predictions.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    stress_predictions.to_csv(OUT_DIR / "stress_predictions.csv", index=False)
    joblib.dump(
        {
            "stage": "research_only_frozen_official_npr_transfer_head",
            "model": model,
            "head_threshold": head_threshold,
            "head_selection": selected_head,
            "fusion_npr_weight": fusion_weight,
            "fusion_threshold": fusion_threshold,
            "fusion_selection": fusion_selected,
            "embedding_dimension": 512,
            "validation_scopes": VALID_SCOPES,
            "test_used_for_selection": False,
            "upstream_license_confirmed": False,
        },
        OUT_DIR / "research_bundle.joblib",
    )

    two_class = test_metrics[test_metrics["eval_scope"].isin(TWO_CLASS_TEST_SCOPES)].copy()
    comparison = (
        two_class.groupby("method_id")
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            mean_roc_auc=("roc_auc", "mean"),
            max_real_fpr=("real_false_positive_rate", "max"),
        )
        .reset_index()
    )
    unknown = test_metrics[test_metrics["eval_scope"].isin(UNKNOWN_GENERATED_SCOPES)].copy()
    summary = {
        "status": "completed",
        "protocol": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "test_rows": int(len(test)),
            "stress_rows": int(len(stress)),
            "validation_scopes": VALID_SCOPES,
            "test_used_for_model_or_threshold_selection": False,
            "selection_formula": (
                "min_balanced_accuracy + 0.50*mean_balanced_accuracy "
                "- 0.35*max_real_fpr - 0.10*std_balanced_accuracy"
            ),
        },
        "selected_head": selected_head,
        "selected_fusion": fusion_selected,
        "two_class_test_aggregate": json_records(comparison),
        "unknown_generated_test_metrics": json_records(unknown),
        "validation_scope_metrics": json_records(validation_metrics),
        "test_scope_metrics": json_records(test_metrics),
        "stress_scope_metrics": json_records(stress_metrics),
        "deployment_decision": (
            "Research only. The frozen NPR representation and its fusion are evaluated as alternative "
            "evidence channels, but the upstream snapshot has no explicit license and the untouched-test "
            "results must show a material, safe gain before any website replacement."
        ),
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (OUT_DIR / "summary.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
