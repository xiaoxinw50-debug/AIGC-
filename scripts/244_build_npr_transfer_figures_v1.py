#!/usr/bin/env python3
"""Build publication-style figures for the NPR transfer and review-guard study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs/tables"
FIGURE_DIR = ROOT / "outputs/figures/npr_transfer_v1"

BURGUNDY = "#7A1F1F"
RED = "#B84A3A"
GOLD = "#C79A52"
SLATE = "#4E6572"
PALE = "#D8D1C7"
INK = "#231F20"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def cross_domain_comparison() -> Path:
    transfer = pd.read_csv(TABLE_DIR / "npr_transfer_test_scope_metrics_v1.csv")
    official = pd.read_csv(TABLE_DIR / "official_npr_test_scope_metrics_v1.csv")
    official = official[official["policy_id"].eq("validation_selected_global_threshold")].copy()
    official["method_id"] = "official_npr_zero_shot"
    combined = pd.concat([transfer, official], ignore_index=True, sort=False)
    methods = [
        ("three_domain_baseline", "Current detector", BURGUNDY),
        ("official_npr_zero_shot", "Official NPR zero-shot", PALE),
        ("validation_selected_npr_transfer_head", "Frozen NPR + new head", SLATE),
        ("validation_selected_baseline_npr_head_fusion", "Validation-selected fusion", GOLD),
    ]
    scopes = [
        ("community_generator_holdout_test", "Community"),
        ("ntire_standard_adapt_test", "NTIRE standard"),
        ("ntire_hard_external_test", "NTIRE hard"),
        ("aigen2026_official_external_test", "AIGen2026"),
    ]
    x = np.arange(len(scopes))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), sharex=True)
    for method_index, (method_id, label, color) in enumerate(methods):
        subset = combined[combined["method_id"].eq(method_id)].set_index("eval_scope")
        offset = (method_index - 1.5) * width
        for ax, metric in zip(axes, ["balanced_accuracy", "real_false_positive_rate"]):
            values = [float(subset.loc[scope, metric]) for scope, _ in scopes]
            ax.bar(x + offset, values, width, label=label, color=color)
    axes[0].set_title("Balanced accuracy", loc="left", weight="bold")
    axes[1].set_title("Real-image false-positive rate", loc="left", weight="bold")
    for ax in axes:
        ax.set_xticks(x, [label for _, label in scopes], rotation=15, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc="lower center")
    fig.suptitle(
        "NPR is complementary on some domains but does not replace the current detector",
        x=0.055,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.91], w_pad=2.5)
    return save(fig, "npr_cross_domain_comparison.png")


def unknown_generator_recall() -> Path:
    transfer = pd.read_csv(TABLE_DIR / "npr_transfer_test_scope_metrics_v1.csv")
    official = pd.read_csv(TABLE_DIR / "official_npr_test_scope_metrics_v1.csv")
    official = official[official["policy_id"].eq("validation_selected_global_threshold")].copy()
    official["method_id"] = "official_npr_zero_shot"
    combined = pd.concat([transfer, official], ignore_index=True, sort=False)
    scopes = [("qwen_unseen_generator_test", "Qwen"), ("safeimg_external_test", "SafeIMG")]
    methods = [
        ("three_domain_baseline", "Current detector", BURGUNDY),
        ("official_npr_zero_shot", "Official NPR zero-shot", PALE),
        ("validation_selected_npr_transfer_head", "Frozen NPR + new head", SLATE),
        ("validation_selected_baseline_npr_head_fusion", "Probability fusion", GOLD),
    ]
    x = np.arange(len(scopes))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for method_index, (method_id, label, color) in enumerate(methods):
        subset = combined[combined["method_id"].eq(method_id)].set_index("eval_scope")
        values = [float(subset.loc[scope, "recall_generated"]) for scope, _ in scopes]
        bars = ax.bar(x + (method_index - 1.5) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, [label for _, label in scopes])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Generated-image recall")
    ax.set_title("Low-level NPR evidence recovers many SafeIMG misses", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    return save(fig, "npr_unknown_generator_recall.png")


def review_guard_tradeoff() -> Path:
    frontier = pd.read_csv(TABLE_DIR / "npr_transfer_review_frontier_v1.csv")
    policies = pd.read_csv(TABLE_DIR / "npr_transfer_review_selected_policies_v1.csv")
    test = pd.read_csv(TABLE_DIR / "npr_transfer_review_test_metrics_v1.csv")
    selected = test[test["policy_id"].eq("npr_transfer_review_cap_03pct")].copy()
    scope_order = [
        ("community_generator_holdout_test", "Community"),
        ("aigen2026_official_external_test", "AIGen2026"),
        ("ntire_standard_adapt_test", "NTIRE standard"),
        ("ntire_hard_external_test", "NTIRE hard"),
        ("qwen_unseen_generator_test", "Qwen"),
        ("safeimg_external_test", "SafeIMG"),
    ]
    selected = selected.set_index("eval_scope")
    rescue = [float(selected.loc[scope, "generated_false_negative_review_rate"]) for scope, _ in scope_order]
    coverage = [float(selected.loc[scope, "auto_coverage"]) for scope, _ in scope_order]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3))
    axes[0].plot(
        frontier["max_real_incremental_review_rate"],
        frontier["total_generated_false_negative_rescue_rate"],
        color=SLATE,
        linewidth=1.7,
    )
    axes[0].scatter(
        policies["max_real_incremental_review_rate"],
        policies["total_generated_false_negative_rescue_rate"],
        c=[GOLD, BURGUNDY, RED],
        s=58,
        zorder=3,
    )
    for _, row in policies.iterrows():
        axes[0].annotate(
            f"{int(row['real_review_cap'] * 100)}% cap",
            (row["max_real_incremental_review_rate"], row["total_generated_false_negative_rescue_rate"]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_xlabel("Maximum validation real-image added-review rate")
    axes[0].set_ylabel("Validation false-negative rescue rate")
    axes[0].set_title("Review guard selected without test tuning", loc="left", weight="bold")
    axes[0].grid(alpha=0.16)

    x = np.arange(len(scope_order))
    width = 0.36
    axes[1].bar(x - width / 2, rescue, width, color=BURGUNDY, label="Misses routed to review")
    axes[1].bar(x + width / 2, coverage, width, color=SLATE, label="Remaining auto coverage")
    axes[1].set_xticks(x, [label for _, label in scope_order], rotation=18, ha="right")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("3% validation-cap policy on untouched tests", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.16)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "NPR is safer as a second-opinion review channel than as an automatic verdict",
        x=0.055,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=2.7)
    return save(fig, "npr_review_guard_tradeoff.png")


def reencoding_stress() -> Path:
    test = pd.read_csv(TABLE_DIR / "npr_transfer_test_scope_metrics_v1.csv")
    test = test[
        test["method_id"].eq("validation_selected_npr_transfer_head")
        & test["eval_scope"].eq("aigen2026_official_external_test")
    ]
    stress = pd.read_csv(TABLE_DIR / "npr_transfer_stress_scope_metrics_v1.csv")
    rows = pd.concat([test, stress], ignore_index=True)
    rows["condition_label"] = rows["eval_scope"].map(
        {
            "aigen2026_official_external_test": "Original files",
            "aigen2026_jpeg_q90_stress_test": "JPEG Q90",
            "aigen2026_resize75_jpeg85_stress_test": "Resize 75% + JPEG85",
        }
    )
    order = ["Original files", "JPEG Q90", "Resize 75% + JPEG85"]
    rows = rows.set_index("condition_label").reindex(order)
    x = np.arange(len(order))
    width = 0.25
    metrics = [
        ("balanced_accuracy", "Balanced accuracy", BURGUNDY),
        ("recall_generated", "Generated recall", SLATE),
        ("real_false_positive_rate", "Real-image FPR", GOLD),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for index, (column, label, color) in enumerate(metrics):
        values = rows[column].to_numpy(dtype=float)
        bars = ax.bar(x + (index - 1) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1.02)
    ax.set_title("Re-encoding weakens the adapted NPR signal", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    return save(fig, "npr_reencoding_stress.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    style()
    paths = [
        cross_domain_comparison(),
        unknown_generator_recall(),
        review_guard_tradeoff(),
        reencoding_stress(),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
