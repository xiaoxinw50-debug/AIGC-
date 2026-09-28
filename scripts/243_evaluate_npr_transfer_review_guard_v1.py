#!/usr/bin/env python3
"""Evaluate NPR transfer-head disagreement as a validation-capped review guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_DIR = ROOT / "outputs/reports/npr_transfer_head_v1"
BASE_VALID = ROOT / "outputs/reports/npr_fusion_v1/validation_predictions.csv"
BASE_TEST = ROOT / "outputs/reports/three_domain_moe_v1/predictions.csv"
TABLE_DIR = ROOT / "outputs/tables"
OUT_DIR = ROOT / "outputs/reports/npr_transfer_review_guard_v1"
VALID_SCOPES = [
    "community_validation",
    "legacy_validation",
    "ntire_standard_adapt_validation",
    "aigen2026_adapt_validation",
]
REAL_REVIEW_CAPS = [0.01, 0.03, 0.05]
SELECTED_CAP = 0.03


def attach_review_flags() -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = pd.read_csv(TRANSFER_DIR / "validation_predictions.csv", low_memory=False)
    base_valid = pd.read_csv(BASE_VALID, low_memory=False)
    validation = validation.merge(
        base_valid[["experiment_sample_id", "eval_scope", "base_needs_review"]],
        on=["experiment_sample_id", "eval_scope"],
        how="left",
        validate="one_to_one",
    )
    test = pd.read_csv(TRANSFER_DIR / "test_predictions.csv", low_memory=False)
    base_test = pd.read_csv(BASE_TEST, low_memory=False)
    test = test.merge(
        base_test[["experiment_sample_id", "eval_scope", "needs_review_validation_selected"]],
        on=["experiment_sample_id", "eval_scope"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"needs_review_validation_selected": "base_needs_review"})
    if validation["base_needs_review"].isna().any() or test["base_needs_review"].isna().any():
        raise RuntimeError("Missing baseline review flags")
    validation["base_needs_review"] = validation["base_needs_review"].astype(bool)
    test["base_needs_review"] = test["base_needs_review"].astype(bool)
    return validation, test


def added_review_mask(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return (
        ~frame["base_needs_review"].to_numpy(dtype=bool)
        & (frame["three_domain_probability"].to_numpy(dtype=float) < 0.5)
        & (frame["npr_transfer_probability"].to_numpy(dtype=float) >= threshold)
    )


def scope_review_metrics(frame: pd.DataFrame, threshold: float, policy_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, part in frame.groupby("eval_scope", sort=False):
        added = added_review_mask(part, threshold)
        base_review = part["base_needs_review"].to_numpy(dtype=bool)
        combined = base_review | added
        y = part["label"].to_numpy(dtype=int)
        probability = part["three_domain_probability"].to_numpy(dtype=float)
        real = y == 0
        generated = y == 1
        generated_false_negative = generated & (probability < 0.5) & ~base_review
        auto = ~combined
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "eval_scope": scope,
            "npr_review_threshold": float(threshold),
            "n": int(len(part)),
            "base_review_count": int(base_review.sum()),
            "base_review_rate": float(base_review.mean()),
            "incremental_review_count": int(added.sum()),
            "incremental_review_rate": float(added.mean()),
            "combined_review_count": int(combined.sum()),
            "combined_review_rate": float(combined.mean()),
            "auto_coverage": float(auto.mean()),
            "real_count": int(real.sum()),
            "real_incremental_review_count": int((added & real).sum()),
            "real_incremental_review_rate": (
                float((added & real).sum() / real.sum()) if real.any() else None
            ),
            "generated_false_negative_count": int(generated_false_negative.sum()),
            "generated_false_negative_routed_to_review": int((added & generated_false_negative).sum()),
            "generated_false_negative_review_rate": (
                float((added & generated_false_negative).sum() / generated_false_negative.sum())
                if generated_false_negative.any()
                else None
            ),
        }
        if auto.any():
            auto_y = y[auto]
            auto_probability = probability[auto]
            auto_prediction = (auto_probability >= 0.5).astype(int)
            row["auto_generated_recall"] = float(recall_score(auto_y, auto_prediction, zero_division=0))
            if len(set(auto_y.tolist())) == 2:
                tn, fp, _, _ = confusion_matrix(auto_y, auto_prediction, labels=[0, 1]).ravel()
                row["auto_balanced_accuracy"] = float(balanced_accuracy_score(auto_y, auto_prediction))
                row["auto_real_false_positive_rate"] = float(fp / max(tn + fp, 1))
                row["auto_roc_auc"] = float(roc_auc_score(auto_y, auto_probability))
            else:
                row["auto_balanced_accuracy"] = None
                row["auto_real_false_positive_rate"] = None
                row["auto_roc_auc"] = None
        else:
            row["auto_generated_recall"] = None
            row["auto_balanced_accuracy"] = None
            row["auto_real_false_positive_rate"] = None
            row["auto_roc_auc"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_candidates(validation: pd.DataFrame) -> np.ndarray:
    eligible_real = validation[
        validation["label"].eq(0)
        & ~validation["base_needs_review"]
        & validation["three_domain_probability"].lt(0.5)
    ]["npr_transfer_probability"].to_numpy(dtype=float)
    quantiles = np.quantile(eligible_real, np.linspace(0.80, 1.0, 101)) if len(eligible_real) else []
    return np.unique(np.concatenate([np.linspace(0.50, 0.999, 101), np.asarray(quantiles)]))


def build_frontier(validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(validation):
        metrics = scope_review_metrics(validation, float(threshold), "candidate")
        selected = metrics[metrics["eval_scope"].isin(VALID_SCOPES)].copy()
        real_rates = selected["real_incremental_review_rate"].dropna().astype(float)
        rescue_rates = selected["generated_false_negative_review_rate"].dropna().astype(float)
        total_fn = int(selected["generated_false_negative_count"].sum())
        total_rescued = int(selected["generated_false_negative_routed_to_review"].sum())
        rows.append(
            {
                "threshold": float(threshold),
                "max_real_incremental_review_rate": float(real_rates.max()),
                "mean_real_incremental_review_rate": float(real_rates.mean()),
                "total_generated_false_negative_count": total_fn,
                "total_generated_false_negative_rescued": total_rescued,
                "total_generated_false_negative_rescue_rate": float(total_rescued / max(total_fn, 1)),
                "mean_scope_generated_false_negative_rescue_rate": float(rescue_rates.mean()),
                "incremental_review_count": int(selected["incremental_review_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_policies(frontier: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cap in REAL_REVIEW_CAPS:
        feasible = frontier[frontier["max_real_incremental_review_rate"] <= cap + 1e-12].copy()
        if feasible.empty:
            raise RuntimeError(f"No feasible threshold under real review cap {cap}")
        selected = feasible.sort_values(
            [
                "total_generated_false_negative_rescue_rate",
                "mean_scope_generated_false_negative_rescue_rate",
                "mean_real_incremental_review_rate",
                "threshold",
            ],
            ascending=[False, False, True, False],
        ).iloc[0].to_dict()
        selected["real_review_cap"] = cap
        selected["policy_id"] = f"npr_transfer_review_cap_{int(cap * 100):02d}pct"
        rows.append(selected)
    return pd.DataFrame(rows)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    validation, test = attach_review_flags()
    frontier = build_frontier(validation)
    policies = select_policies(frontier)
    validation_metrics: list[pd.DataFrame] = []
    test_metrics: list[pd.DataFrame] = []
    for row in policies.to_dict(orient="records"):
        threshold = float(row["threshold"])
        policy_id = str(row["policy_id"])
        validation_metrics.append(scope_review_metrics(validation, threshold, policy_id))
        test_metrics.append(scope_review_metrics(test, threshold, policy_id))
    validation_metrics_frame = pd.concat(validation_metrics, ignore_index=True)
    test_metrics_frame = pd.concat(test_metrics, ignore_index=True)

    selected = policies[np.isclose(policies["real_review_cap"], SELECTED_CAP)].iloc[0]
    selected_threshold = float(selected["threshold"])
    selected_policy_id = str(selected["policy_id"])
    validation_output = validation.copy()
    test_output = test.copy()
    for frame in [validation_output, test_output]:
        frame["npr_transfer_added_review"] = added_review_mask(frame, selected_threshold).astype(int)
        frame["combined_review"] = (
            frame["base_needs_review"].to_numpy(dtype=bool)
            | frame["npr_transfer_added_review"].to_numpy(dtype=bool)
        ).astype(int)

    frontier.to_csv(TABLE_DIR / "npr_transfer_review_frontier_v1.csv", index=False)
    policies.to_csv(TABLE_DIR / "npr_transfer_review_selected_policies_v1.csv", index=False)
    validation_metrics_frame.to_csv(
        TABLE_DIR / "npr_transfer_review_validation_metrics_v1.csv", index=False
    )
    test_metrics_frame.to_csv(TABLE_DIR / "npr_transfer_review_test_metrics_v1.csv", index=False)
    validation_output.to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    test_output.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    selected_test = test_metrics_frame[test_metrics_frame["policy_id"].eq(selected_policy_id)].copy()
    summary = {
        "status": "completed",
        "selection_rule": (
            "Among thresholds whose maximum validation real-image incremental review rate is at most "
            "the policy cap, maximize total baseline false-negative rescue; break ties by mean scope rescue, "
            "lower mean real review, and higher threshold."
        ),
        "selected_operational_cap": SELECTED_CAP,
        "selected_policy": json_records(policies[policies["policy_id"].eq(selected_policy_id)])[0],
        "all_selected_policies": json_records(policies),
        "selected_policy_test_metrics": json_records(selected_test),
        "deployment_decision": (
            "Research-only review guard. It never changes an automatic real decision directly to generated; "
            "it only adds a review recommendation. SafeIMG has no matched real controls and the NPR upstream "
            "snapshot has no explicit license, so this guard is not enabled on the public website."
        ),
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (OUT_DIR / "summary.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
