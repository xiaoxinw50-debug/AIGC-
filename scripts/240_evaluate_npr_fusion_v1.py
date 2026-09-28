#!/usr/bin/env python3
"""Test whether official NPR adds safe value to the three-domain detector."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
NPR_VALID = ROOT / "outputs/reports/official_npr_v1/validation_predictions.csv"
NPR_TEST = ROOT / "outputs/reports/official_npr_v1/test_predictions.csv"
BASE_TEST = ROOT / "outputs/reports/three_domain_moe_v1/predictions.csv"
BASE_BUNDLE = ROOT / "outputs/reports/three_domain_moe_v1/model_bundle.joblib"
OLD_INDEX = ROOT / "outputs/tables/clip_binary_embedding_manifest_v1.csv"
OLD_EMBEDDINGS = ROOT / "outputs/features/clip_binary_v1/embeddings.npy"
OLD_STATUS = ROOT / "outputs/features/clip_binary_v1/status.npy"
NEW_INDEX = ROOT / "outputs/tables/clip_aigenimages2026_manifest_v1.csv"
NEW_EMBEDDINGS = ROOT / "outputs/features/clip_aigenimages2026_v1/embeddings.npy"
NEW_STATUS = ROOT / "outputs/features/clip_aigenimages2026_v1/status.npy"
OLD_FEATURES = ROOT / "data/processed/features/modern_ai_detection_enhanced_meta_features_v1.csv"
NEW_FEATURES = ROOT / "data/processed/features/aigenimages2026_enhanced_meta_features_v1.csv"
OUT_DIR = ROOT / "outputs/reports/npr_fusion_v1"
TABLE_DIR = ROOT / "outputs/tables"
RANDOM_STATE = 42

SELECTION_SCOPES = [
    "community_validation",
    "ntire_standard_adapt_validation",
    "aigen2026_adapt_validation",
]
VALID_TO_TEST_SCOPE = {
    "community_validation": "community_generator_holdout_test",
    "ntire_standard_adapt_validation": "ntire_standard_adapt_test",
    "aigen2026_adapt_validation": "aigen2026_official_external_test",
}


def metric_block(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    labels = set(y_true.tolist())
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "real_count": int((y_true == 0).sum()),
        "generated_count": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)) if len(labels) == 2 else None,
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


def selective_metric_block(y_true: np.ndarray, probability: np.ndarray, review: np.ndarray) -> dict[str, Any]:
    auto = ~review
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "auto_count": int(auto.sum()),
        "review_count": int(review.sum()),
        "auto_coverage": float(auto.mean()),
        "review_rate": float(review.mean()),
    }
    if not auto.any():
        return result
    auto_metrics = metric_block(y_true[auto], probability[auto])
    result.update({f"auto_{key}": value for key, value in auto_metrics.items()})
    return result


def read_feature_lookup(feature_names: list[str]) -> pd.DataFrame:
    columns = ["experiment_sample_id", *feature_names]
    old = pd.read_csv(OLD_FEATURES, usecols=columns, low_memory=False)
    new = pd.read_csv(NEW_FEATURES, usecols=columns, low_memory=False)
    combined = pd.concat([old, new], ignore_index=True)
    if combined["experiment_sample_id"].duplicated().any():
        raise RuntimeError("Duplicate feature ids across modern and AIGen tables")
    return combined.set_index("experiment_sample_id")


def embedding_matrix(sample_ids: pd.Series) -> np.ndarray:
    old_index = pd.read_csv(OLD_INDEX).set_index("experiment_sample_id")
    new_index = pd.read_csv(NEW_INDEX).set_index("experiment_sample_id")
    old_vectors = np.load(OLD_EMBEDDINGS, mmap_mode="r")
    new_vectors = np.load(NEW_EMBEDDINGS, mmap_mode="r")
    old_status = np.load(OLD_STATUS, mmap_mode="r")
    new_status = np.load(NEW_STATUS, mmap_mode="r")
    dimension = int(old_vectors.shape[1])
    result = np.empty((len(sample_ids), dimension), dtype="float32")
    for output_row, sample_id in enumerate(sample_ids.astype(str)):
        if sample_id in old_index.index:
            source_row = int(old_index.loc[sample_id, "embedding_row"])
            if int(old_status[source_row]) != 1:
                raise RuntimeError(f"Incomplete old embedding: {sample_id}")
            result[output_row] = old_vectors[source_row]
        elif sample_id in new_index.index:
            source_row = int(new_index.loc[sample_id, "embedding_row"])
            if int(new_status[source_row]) != 1:
                raise RuntimeError(f"Incomplete AIGen embedding: {sample_id}")
            result[output_row] = new_vectors[source_row]
        else:
            raise KeyError(f"Missing embedding id: {sample_id}")
    return result


def add_three_domain_validation_probability(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(BASE_BUNDLE)
    result = frame.copy()
    embeddings = embedding_matrix(result["experiment_sample_id"])
    features = read_feature_lookup(bundle["router_feature_names"]).reindex(result["experiment_sample_id"])
    if features.isna().any().any():
        missing = features.index[features.isna().any(axis=1)].tolist()[:5]
        raise RuntimeError(f"Missing router features: {missing}")
    router_x = features[bundle["router_feature_names"]].to_numpy(dtype="float32")
    router_probability = bundle["router"].predict_proba(router_x)
    router_domain = bundle["router"].classes_[np.argmax(router_probability, axis=1)].astype(int)
    expert_probability = np.column_stack(
        [
            bundle["base_model"].predict_proba(embeddings)[:, 1],
            bundle["ntire_model"].predict_proba(embeddings)[:, 1],
            bundle["aigen_model"].predict_proba(embeddings)[:, 1],
        ]
    )
    result["router_domain"] = router_domain
    result["router_confidence"] = router_probability.max(axis=1)
    result["probability_generated"] = expert_probability[np.arange(len(result)), router_domain]
    result["base_needs_review"] = (
        np.abs(result["probability_generated"].to_numpy(dtype=float) - 0.5)
        < float(bundle["review_class_margin"])
    ).astype(int)
    return result


def evaluate_by_scope(frame: pd.DataFrame, probability_column: str, method_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, part in frame.groupby("eval_scope", sort=False):
        rows.append(
            {
                "method_id": method_id,
                "eval_scope": scope,
                **metric_block(
                    part["label"].to_numpy(dtype=int),
                    part[probability_column].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def select_fusion_weight(validation: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for weight in np.linspace(0.0, 0.50, 21):
        probability = (
            (1.0 - weight) * validation["probability_generated"].to_numpy(dtype=float)
            + weight * validation["npr_calibrated_probability"].to_numpy(dtype=float)
        )
        candidate = validation.copy()
        candidate["fusion_probability"] = probability
        metrics = evaluate_by_scope(candidate, "fusion_probability", f"npr_weight_{weight:.3f}")
        metrics = metrics[metrics["eval_scope"].isin(SELECTION_SCOPES)]
        ba = metrics["balanced_accuracy"].astype(float)
        fpr = metrics["real_false_positive_rate"].astype(float)
        rows.append(
            {
                "npr_weight": float(weight),
                "min_balanced_accuracy": float(ba.min()),
                "mean_balanced_accuracy": float(ba.mean()),
                "max_real_fpr": float(fpr.max()),
                "selection_score": float(ba.min() + ba.mean() - 0.5 * fpr.max()),
            }
        )
    table = pd.DataFrame(rows)
    baseline = table[table["npr_weight"].eq(0.0)].iloc[0]
    feasible = table[
        (table["max_real_fpr"] <= float(baseline["max_real_fpr"]) + 0.02)
        & (table["min_balanced_accuracy"] >= float(baseline["min_balanced_accuracy"]) - 0.01)
    ]
    selected = feasible.sort_values(
        ["selection_score", "min_balanced_accuracy", "mean_balanced_accuracy"],
        ascending=False,
    ).iloc[0]
    return float(selected["npr_weight"]), table


def select_review_threshold(validation: pd.DataFrame, real_review_cap: float = 0.05) -> float:
    thresholds: list[float] = []
    for scope in SELECTION_SCOPES:
        part = validation[
            validation["eval_scope"].eq(scope)
            & validation["label"].eq(0)
            & validation["base_needs_review"].eq(0)
            & (validation["probability_generated"] < 0.5)
        ]
        if part.empty:
            continue
        thresholds.append(float(part["npr_logit"].quantile(1.0 - real_review_cap, interpolation="higher")))
    if not thresholds:
        return float("inf")
    return max(thresholds)


def evaluate_review_policy(
    frame: pd.DataFrame,
    review_threshold: float,
    method_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    base_review = result["base_needs_review"].astype(bool).to_numpy()
    npr_added_review = (
        ~base_review
        & (result["probability_generated"].to_numpy(dtype=float) < 0.5)
        & (result["npr_logit"].to_numpy(dtype=float) >= review_threshold)
    )
    combined_review = base_review | npr_added_review
    result["npr_added_review"] = npr_added_review.astype(int)
    result["combined_review"] = combined_review.astype(int)
    rows: list[dict[str, Any]] = []
    for scope, part in result.groupby("eval_scope", sort=False):
        y = part["label"].to_numpy(dtype=int)
        added = part["npr_added_review"].astype(bool).to_numpy()
        generated_false_negative = (y == 1) & (part["probability_generated"].to_numpy(dtype=float) < 0.5)
        real = y == 0
        rows.append(
            {
                "method_id": method_id,
                "eval_scope": scope,
                "npr_review_logit_threshold": review_threshold,
                "incremental_review_count": int(added.sum()),
                "incremental_review_rate": float(added.mean()),
                "real_incremental_review_count": int((added & real).sum()),
                "real_incremental_review_rate": float((added & real).sum() / max(real.sum(), 1)),
                "generated_false_negative_count": int(generated_false_negative.sum()),
                "generated_false_negative_routed_to_review": int((added & generated_false_negative).sum()),
                "generated_false_negative_review_rate": float(
                    (added & generated_false_negative).sum() / max(generated_false_negative.sum(), 1)
                ),
                **selective_metric_block(
                    y,
                    part["probability_generated"].to_numpy(dtype=float),
                    part["combined_review"].astype(bool).to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows), result


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    npr_valid = pd.read_csv(NPR_VALID, low_memory=False)
    validation = add_three_domain_validation_probability(npr_valid)
    calibration_pool = validation[validation["eval_scope"].isin(SELECTION_SCOPES)].copy()
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    calibrator.fit(
        calibration_pool[["npr_logit"]].to_numpy(dtype=float),
        calibration_pool["label"].to_numpy(dtype=int),
    )
    validation["npr_calibrated_probability"] = calibrator.predict_proba(
        validation[["npr_logit"]].to_numpy(dtype=float)
    )[:, 1]

    selected_weight, fusion_candidates = select_fusion_weight(validation)
    validation["selected_fusion_probability"] = (
        (1.0 - selected_weight) * validation["probability_generated"]
        + selected_weight * validation["npr_calibrated_probability"]
    )
    validation_metrics = pd.concat(
        [
            evaluate_by_scope(validation, "probability_generated", "three_domain_baseline"),
            evaluate_by_scope(validation, "selected_fusion_probability", "validation_selected_npr_fusion"),
        ],
        ignore_index=True,
    )

    review_threshold = select_review_threshold(validation)
    validation_review_metrics, validation_with_review = evaluate_review_policy(
        validation,
        review_threshold,
        "three_domain_plus_npr_review_guard",
    )

    npr_test = pd.read_csv(NPR_TEST, low_memory=False)
    npr_test = npr_test[npr_test["condition"].eq("original")].copy()
    base_test = pd.read_csv(BASE_TEST, low_memory=False)
    test = base_test.merge(
        npr_test[["experiment_sample_id", "eval_scope", "npr_logit", "npr_probability"]],
        on=["experiment_sample_id", "eval_scope"],
        how="inner",
        validate="one_to_one",
    )
    if len(test) != len(base_test):
        raise RuntimeError(f"NPR/base test merge mismatch: {len(test)} vs {len(base_test)}")
    test["base_needs_review"] = test["needs_review_validation_selected"].astype(int)
    test["npr_calibrated_probability"] = calibrator.predict_proba(
        test[["npr_logit"]].to_numpy(dtype=float)
    )[:, 1]
    test["selected_fusion_probability"] = (
        (1.0 - selected_weight) * test["probability_generated"]
        + selected_weight * test["npr_calibrated_probability"]
    )
    test_metrics = pd.concat(
        [
            evaluate_by_scope(test, "probability_generated", "three_domain_baseline"),
            evaluate_by_scope(test, "selected_fusion_probability", "validation_selected_npr_fusion"),
        ],
        ignore_index=True,
    )
    test_review_metrics, test_with_review = evaluate_review_policy(
        test,
        review_threshold,
        "three_domain_plus_npr_review_guard",
    )

    fusion_candidates.to_csv(TABLE_DIR / "npr_fusion_validation_candidates_v1.csv", index=False)
    validation_metrics.to_csv(TABLE_DIR / "npr_fusion_validation_scope_metrics_v1.csv", index=False)
    test_metrics.to_csv(TABLE_DIR / "npr_fusion_test_scope_metrics_v1.csv", index=False)
    validation_review_metrics.to_csv(TABLE_DIR / "npr_review_guard_validation_metrics_v1.csv", index=False)
    test_review_metrics.to_csv(TABLE_DIR / "npr_review_guard_test_metrics_v1.csv", index=False)
    validation_with_review.to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    test_with_review.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "stage": "research_only_npr_fusion",
            "npr_platt_calibrator": calibrator,
            "selected_npr_weight": selected_weight,
            "npr_review_logit_threshold": review_threshold,
            "selection_scopes": SELECTION_SCOPES,
            "upstream_license_confirmed": False,
        },
        OUT_DIR / "research_bundle.joblib",
    )

    safeimg = test_review_metrics[test_review_metrics["eval_scope"].eq("safeimg_external_test")].iloc[0]
    summary = {
        "status": "completed",
        "selected_npr_fusion_weight": selected_weight,
        "platt_coefficient": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "npr_review_logit_threshold": review_threshold,
        "validation_selection_scopes": SELECTION_SCOPES,
        "safeimg_incremental_review_rate": float(safeimg["incremental_review_rate"]),
        "safeimg_false_negative_review_rate": float(safeimg["generated_false_negative_review_rate"]),
        "fusion_validation_candidates": json_records(fusion_candidates),
        "validation_scope_metrics": json_records(validation_metrics),
        "test_scope_metrics": json_records(test_metrics),
        "validation_review_metrics": json_records(validation_review_metrics),
        "test_review_metrics": json_records(test_review_metrics),
        "deployment_decision": (
            "Do not deploy NPR probability fusion or weights. The upstream snapshot lacks an explicit license, "
            "and validation must show non-zero fusion weight plus safe real-image behavior before deployment. "
            "The review guard remains research-only because SafeIMG has no matched real controls."
        ),
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (OUT_DIR / "summary.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
