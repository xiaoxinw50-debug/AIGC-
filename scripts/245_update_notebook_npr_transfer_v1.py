#!/usr/bin/env python3
"""Insert the completed NPR alternative-evidence study into the project notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/04_execution/AIGC标识治理_实验复现与结论总览.ipynb"
MARKER = "#### 6.5.11 NPR 低层痕迹实验：从零样本到第二意见通道"
NEXT_SECTION = "## 7. 开放场景、阈值策略与治理队列"
FIGURE_DIR = ROOT / "outputs/figures/npr_transfer_v1"


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip() + "\n")}


def table_output(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "output_type": "display_data",
            "metadata": {},
            "data": {
                "text/plain": [frame.to_string(index=False)],
                "text/html": [frame.to_html(index=False, border=0, classes="dataframe")],
            },
        }
    ]


def code_table(code: str, frame: pd.DataFrame, count: int) -> dict:
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": table_output(frame),
        "source": lines(code.strip() + "\n"),
    }


def code_stdout(code: str, output: str, count: int) -> dict:
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": [{"name": "stdout", "output_type": "stream", "text": lines(output.rstrip() + "\n")}],
        "source": lines(code.strip() + "\n"),
    }


def code_image(path: Path, count: int, width: int = 1150) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    relative = path.relative_to(ROOT)
    code = f"from IPython.display import Image, display\ndisplay(Image(filename={str(relative)!r}, width={width}))"
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": [
            {
                "output_type": "display_data",
                "metadata": {"image/png": {"alt": path.stem.replace("_", " "), "width": width}},
                "data": {"image/png": encoded, "text/plain": [f"<IPython.core.display.Image object: {relative}>"]},
            }
        ],
        "source": lines(code + "\n"),
    }


def metric_view(frame: pd.DataFrame, scopes: list[str]) -> pd.DataFrame:
    method_names = {
        "three_domain_baseline": "现有三域模型",
        "validation_selected_npr_transfer_head": "NPR 冻结骨干+迁移头",
        "validation_selected_baseline_npr_head_fusion": "验证集选择的概率融合",
    }
    scope_names = {
        "community_generator_holdout_test": "Community 未见生成器",
        "legacy_comparability_test": "Legacy 可比测试",
        "native_export_test": "平台原生导出小样本",
        "ntire_standard_adapt_test": "NTIRE 标准测试",
        "ntire_hard_external_test": "NTIRE 困难外测",
        "aigen2026_official_external_test": "AIGenImages2026 官方外测",
        "qwen_unseen_generator_test": "Qwen 未见生成器",
        "safeimg_external_test": "SafeIMG 外部生成集",
    }
    view = frame[frame["eval_scope"].isin(scopes)].copy()
    view["方法"] = view["method_id"].map(method_names)
    view["测试范围"] = view["eval_scope"].map(scope_names)
    return view[
        [
            "方法",
            "测试范围",
            "n",
            "balanced_accuracy",
            "recall_generated",
            "real_false_positive_rate",
            "roc_auc",
        ]
    ].rename(
        columns={
            "n": "样本数",
            "balanced_accuracy": "平衡准确率",
            "recall_generated": "生成图召回率",
            "real_false_positive_rate": "真实图误报率",
            "roc_auc": "ROC AUC",
        }
    ).round(4)


def main() -> None:
    required = [
        ROOT / "outputs/features/npr_official_embeddings_v1/metadata.json",
        ROOT / "outputs/reports/official_npr_v1/summary.json",
        ROOT / "outputs/reports/npr_transfer_head_v1/summary.json",
        ROOT / "outputs/reports/npr_transfer_review_guard_v1/summary.json",
        ROOT / "outputs/tables/npr_transfer_test_scope_metrics_v1.csv",
        ROOT / "outputs/tables/npr_transfer_stress_scope_metrics_v1.csv",
        ROOT / "outputs/tables/npr_transfer_review_selected_policies_v1.csv",
        ROOT / "outputs/tables/npr_transfer_review_test_metrics_v1.csv",
        ROOT / "outputs/tables/npr_transfer_head_validation_candidates_v1.csv",
        FIGURE_DIR / "npr_cross_domain_comparison.png",
        FIGURE_DIR / "npr_unknown_generator_recall.png",
        FIGURE_DIR / "npr_review_guard_tradeoff.png",
        FIGURE_DIR / "npr_reencoding_stress.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run NPR experiments first. Missing: {missing}")

    embedding_meta = json.loads(required[0].read_text(encoding="utf-8"))
    official_summary = json.loads(required[1].read_text(encoding="utf-8"))
    transfer_summary = json.loads(required[2].read_text(encoding="utf-8"))
    review_summary = json.loads(required[3].read_text(encoding="utf-8"))
    test_metrics = pd.read_csv(required[4])
    stress_metrics = pd.read_csv(required[5])
    review_policies = pd.read_csv(required[6])
    review_test = pd.read_csv(required[7])
    head_candidates = pd.read_csv(required[8])

    split_rows = {item["split"]: item["rows"] for item in embedding_meta["outputs"]}
    protocol = pd.DataFrame(
        [
            {"数据区": "迁移训练集", "样本数": split_rows["train"], "用途": "拟合新线性头", "参与选模": "是"},
            {"数据区": "内部验证集", "样本数": split_rows["valid"], "用途": "正则、权重、阈值与复核策略选择", "参与选模": "是"},
            {"数据区": "原始图独立测试集", "样本数": split_rows["test"], "用途": "一次性外测", "参与选模": "否"},
            {"数据区": "重编码压力集", "样本数": split_rows["stress"], "用途": "JPEG 与缩放传播鲁棒性", "参与选模": "否"},
        ]
    )

    official_metrics = pd.DataFrame(official_summary["test_scope_metrics"])
    official_view = official_metrics[
        official_metrics["policy_id"].eq("validation_selected_global_threshold")
        & official_metrics["eval_scope"].isin(
            [
                "community_generator_holdout_test",
                "ntire_standard_adapt_test",
                "ntire_hard_external_test",
                "aigen2026_official_external_test",
                "qwen_unseen_generator_test",
                "safeimg_external_test",
            ]
        )
    ][
        [
            "eval_scope",
            "n",
            "balanced_accuracy",
            "recall_generated",
            "real_false_positive_rate",
            "roc_auc",
        ]
    ].round(4)

    selected_head = transfer_summary["selected_head"]
    selected_fusion = transfer_summary["selected_fusion"]
    candidate_audit = pd.DataFrame(
        [
            {"检查项": "冻结表征维度", "结果": embedding_meta["dimension"], "说明": "NPR 骨干全局平均池化后 L2 归一化"},
            {"检查项": "头模型候选数", "结果": int(head_candidates[["weight_strategy", "regularization_c"]].drop_duplicates().shape[0]), "说明": "3 种权重方案 × 4 个 C"},
            {"检查项": "阈值候选数", "结果": int(head_candidates["threshold"].nunique()), "说明": "仅用验证集搜索"},
            {"检查项": "最终样本权重", "结果": selected_head["weight_strategy"], "说明": "域与类别双平衡"},
            {"检查项": "最终正则强度 C", "结果": selected_head["regularization_c"], "说明": "逻辑回归"},
            {"检查项": "最终阈值", "结果": selected_head["threshold"], "说明": "测试集未参与选择"},
        ]
    )

    key_scopes = [
        "community_generator_holdout_test",
        "ntire_standard_adapt_test",
        "ntire_hard_external_test",
        "aigen2026_official_external_test",
    ]
    comparison_view = metric_view(test_metrics, key_scopes)
    unknown_view = metric_view(
        test_metrics,
        ["qwen_unseen_generator_test", "safeimg_external_test"],
    )
    stress_view = stress_metrics[
        [
            "eval_scope",
            "n",
            "balanced_accuracy",
            "recall_generated",
            "real_false_positive_rate",
            "roc_auc",
        ]
    ].rename(
        columns={
            "eval_scope": "压力条件",
            "n": "样本数",
            "balanced_accuracy": "平衡准确率",
            "recall_generated": "生成图召回率",
            "real_false_positive_rate": "真实图误报率",
            "roc_auc": "ROC AUC",
        }
    ).round(4)
    original_row = test_metrics[
        test_metrics["method_id"].eq("validation_selected_npr_transfer_head")
        & test_metrics["eval_scope"].eq("aigen2026_official_external_test")
    ][
        [
            "eval_scope",
            "n",
            "balanced_accuracy",
            "recall_generated",
            "real_false_positive_rate",
            "roc_auc",
        ]
    ].rename(
        columns={
            "eval_scope": "压力条件",
            "n": "样本数",
            "balanced_accuracy": "平衡准确率",
            "recall_generated": "生成图召回率",
            "real_false_positive_rate": "真实图误报率",
            "roc_auc": "ROC AUC",
        }
    ).round(4)
    stress_view = pd.concat([original_row, stress_view], ignore_index=True)

    policy_view = review_policies[
        [
            "real_review_cap",
            "threshold",
            "max_real_incremental_review_rate",
            "total_generated_false_negative_rescue_rate",
            "incremental_review_count",
        ]
    ].rename(
        columns={
            "real_review_cap": "验证集真实图复核上限",
            "threshold": "NPR 复核阈值",
            "max_real_incremental_review_rate": "实际最高新增真实图复核率",
            "total_generated_false_negative_rescue_rate": "验证集漏检转复核率",
            "incremental_review_count": "新增复核数",
        }
    ).round(4)
    selected_review = review_test[review_test["policy_id"].eq("npr_transfer_review_cap_03pct")].copy()
    review_scope_names = {
        "community_generator_holdout_test": "Community",
        "aigen2026_official_external_test": "AIGen2026",
        "ntire_standard_adapt_test": "NTIRE 标准",
        "ntire_hard_external_test": "NTIRE 困难",
        "qwen_unseen_generator_test": "Qwen",
        "safeimg_external_test": "SafeIMG",
    }
    selected_review = selected_review[selected_review["eval_scope"].isin(review_scope_names)].copy()
    selected_review["测试范围"] = selected_review["eval_scope"].map(review_scope_names)
    review_view = selected_review[
        [
            "测试范围",
            "incremental_review_rate",
            "real_incremental_review_rate",
            "generated_false_negative_review_rate",
            "auto_coverage",
            "auto_balanced_accuracy",
        ]
    ].rename(
        columns={
            "incremental_review_rate": "新增复核率",
            "real_incremental_review_rate": "真实图新增复核率",
            "generated_false_negative_review_rate": "漏检转复核率",
            "auto_coverage": "剩余自动覆盖率",
            "auto_balanced_accuracy": "自动部分平衡准确率",
        }
    ).round(4)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    existing_start = next((i for i, cell in enumerate(cells) if cell_source(cell).startswith(MARKER)), None)
    if existing_start is not None:
        existing_end = next(
            (i for i in range(existing_start + 1, len(cells)) if cell_source(cells[i]).startswith(NEXT_SECTION)),
            len(cells),
        )
        del cells[existing_start:existing_end]
    insert_at = next(i for i, cell in enumerate(cells) if cell_source(cell).startswith(NEXT_SECTION))
    max_execution = max(
        [cell.get("execution_count") or 0 for cell in cells if cell.get("cell_type") == "code"],
        default=0,
    )
    counter = max_execution + 1

    setup_code = r'''
import json
import pandas as pd

npr_embedding_meta = json.loads((ROOT / "outputs/features/npr_official_embeddings_v1/metadata.json").read_text(encoding="utf-8"))
npr_official_summary = json.loads((ROOT / "outputs/reports/official_npr_v1/summary.json").read_text(encoding="utf-8"))
npr_transfer_summary = json.loads((ROOT / "outputs/reports/npr_transfer_head_v1/summary.json").read_text(encoding="utf-8"))
npr_review_summary = json.loads((ROOT / "outputs/reports/npr_transfer_review_guard_v1/summary.json").read_text(encoding="utf-8"))
npr_transfer_metrics_raw = pd.read_csv(ROOT / "outputs/tables/npr_transfer_test_scope_metrics_v1.csv")
npr_stress_raw = pd.read_csv(ROOT / "outputs/tables/npr_transfer_stress_scope_metrics_v1.csv")
npr_review_policies_raw = pd.read_csv(ROOT / "outputs/tables/npr_transfer_review_selected_policies_v1.csv")
npr_review_test_raw = pd.read_csv(ROOT / "outputs/tables/npr_transfer_review_test_metrics_v1.csv")
npr_head_candidates_raw = pd.read_csv(ROOT / "outputs/tables/npr_transfer_head_validation_candidates_v1.csv")

_npr_split_rows = {item["split"]: item["rows"] for item in npr_embedding_meta["outputs"]}
npr_protocol_v1 = pd.DataFrame([
    {"数据区": "迁移训练集", "样本数": _npr_split_rows["train"], "用途": "拟合新线性头", "参与选模": "是"},
    {"数据区": "内部验证集", "样本数": _npr_split_rows["valid"], "用途": "正则、权重、阈值与复核策略选择", "参与选模": "是"},
    {"数据区": "原始图独立测试集", "样本数": _npr_split_rows["test"], "用途": "一次性外测", "参与选模": "否"},
    {"数据区": "重编码压力集", "样本数": _npr_split_rows["stress"], "用途": "JPEG 与缩放传播鲁棒性", "参与选模": "否"},
])

_npr_official_metrics = pd.DataFrame(npr_official_summary["test_scope_metrics"])
official_npr_zero_shot_metrics = _npr_official_metrics[
    _npr_official_metrics["policy_id"].eq("validation_selected_global_threshold")
    & _npr_official_metrics["eval_scope"].isin([
        "community_generator_holdout_test", "ntire_standard_adapt_test",
        "ntire_hard_external_test", "aigen2026_official_external_test",
        "qwen_unseen_generator_test", "safeimg_external_test",
    ])
][["eval_scope", "n", "balanced_accuracy", "recall_generated", "real_false_positive_rate", "roc_auc"]].round(4)

_npr_selected_head = npr_transfer_summary["selected_head"]
npr_transfer_candidate_audit = pd.DataFrame([
    {"检查项": "冻结表征维度", "结果": npr_embedding_meta["dimension"], "说明": "NPR 骨干全局平均池化后 L2 归一化"},
    {"检查项": "头模型候选数", "结果": int(npr_head_candidates_raw[["weight_strategy", "regularization_c"]].drop_duplicates().shape[0]), "说明": "3 种权重方案 × 4 个 C"},
    {"检查项": "阈值候选数", "结果": int(npr_head_candidates_raw["threshold"].nunique()), "说明": "仅用验证集搜索"},
    {"检查项": "最终样本权重", "结果": _npr_selected_head["weight_strategy"], "说明": "域与类别双平衡"},
    {"检查项": "最终正则强度 C", "结果": _npr_selected_head["regularization_c"], "说明": "逻辑回归"},
    {"检查项": "最终阈值", "结果": _npr_selected_head["threshold"], "说明": "测试集未参与选择"},
])

_npr_method_names = {
    "three_domain_baseline": "现有三域模型",
    "validation_selected_npr_transfer_head": "NPR 冻结骨干+迁移头",
    "validation_selected_baseline_npr_head_fusion": "验证集选择的概率融合",
}
_npr_scope_names = {
    "community_generator_holdout_test": "Community 未见生成器",
    "ntire_standard_adapt_test": "NTIRE 标准测试",
    "ntire_hard_external_test": "NTIRE 困难外测",
    "aigen2026_official_external_test": "AIGenImages2026 官方外测",
    "qwen_unseen_generator_test": "Qwen 未见生成器",
    "safeimg_external_test": "SafeIMG 外部生成集",
}
def _npr_metric_view(scopes):
    view = npr_transfer_metrics_raw[npr_transfer_metrics_raw["eval_scope"].isin(scopes)].copy()
    view["方法"] = view["method_id"].map(_npr_method_names)
    view["测试范围"] = view["eval_scope"].map(_npr_scope_names)
    return view[["方法", "测试范围", "n", "balanced_accuracy", "recall_generated", "real_false_positive_rate", "roc_auc"]].rename(columns={
        "n": "样本数", "balanced_accuracy": "平衡准确率", "recall_generated": "生成图召回率",
        "real_false_positive_rate": "真实图误报率", "roc_auc": "ROC AUC",
    }).round(4)

npr_transfer_cross_domain_metrics = _npr_metric_view([
    "community_generator_holdout_test", "ntire_standard_adapt_test",
    "ntire_hard_external_test", "aigen2026_official_external_test",
])
npr_unknown_generator_metrics = _npr_metric_view(["qwen_unseen_generator_test", "safeimg_external_test"])

_npr_stress_columns = ["eval_scope", "n", "balanced_accuracy", "recall_generated", "real_false_positive_rate", "roc_auc"]
_npr_stress_rename = {
    "eval_scope": "压力条件", "n": "样本数", "balanced_accuracy": "平衡准确率",
    "recall_generated": "生成图召回率", "real_false_positive_rate": "真实图误报率", "roc_auc": "ROC AUC",
}
_npr_original = npr_transfer_metrics_raw[
    npr_transfer_metrics_raw["method_id"].eq("validation_selected_npr_transfer_head")
    & npr_transfer_metrics_raw["eval_scope"].eq("aigen2026_official_external_test")
][_npr_stress_columns].rename(columns=_npr_stress_rename).round(4)
npr_reencoding_stress_metrics = pd.concat([
    _npr_original,
    npr_stress_raw[_npr_stress_columns].rename(columns=_npr_stress_rename).round(4),
], ignore_index=True)

npr_review_guard_validation_policies = npr_review_policies_raw[[
    "real_review_cap", "threshold", "max_real_incremental_review_rate",
    "total_generated_false_negative_rescue_rate", "incremental_review_count",
]].rename(columns={
    "real_review_cap": "验证集真实图复核上限", "threshold": "NPR 复核阈值",
    "max_real_incremental_review_rate": "实际最高新增真实图复核率",
    "total_generated_false_negative_rescue_rate": "验证集漏检转复核率", "incremental_review_count": "新增复核数",
}).round(4)
_npr_review_names = {
    "community_generator_holdout_test": "Community", "aigen2026_official_external_test": "AIGen2026",
    "ntire_standard_adapt_test": "NTIRE 标准", "ntire_hard_external_test": "NTIRE 困难",
    "qwen_unseen_generator_test": "Qwen", "safeimg_external_test": "SafeIMG",
}
_npr_review_selected = npr_review_test_raw[
    npr_review_test_raw["policy_id"].eq("npr_transfer_review_cap_03pct")
    & npr_review_test_raw["eval_scope"].isin(_npr_review_names)
].copy()
_npr_review_selected["测试范围"] = _npr_review_selected["eval_scope"].map(_npr_review_names)
npr_review_guard_test_metrics = _npr_review_selected[[
    "测试范围", "incremental_review_rate", "real_incremental_review_rate",
    "generated_false_negative_review_rate", "auto_coverage", "auto_balanced_accuracy",
]].rename(columns={
    "incremental_review_rate": "新增复核率", "real_incremental_review_rate": "真实图新增复核率",
    "generated_false_negative_review_rate": "漏检转复核率", "auto_coverage": "剩余自动覆盖率",
    "auto_balanced_accuracy": "自动部分平衡准确率",
}).round(4)

npr_reproduction_commands = "\n".join([
    "# PyTorch 2.9.1 environment: extract official NPR frozen embeddings",
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/241_extract_npr_embeddings_v1.py",
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/241_extract_npr_embeddings_v1.py --splits stress",
    "# Project environment: train the transfer head and evaluate the review guard",
    "./.venv/bin/python scripts/242_train_npr_transfer_head_v1.py",
    "./.venv/bin/python scripts/243_evaluate_npr_transfer_review_guard_v1.py",
    "./.venv/bin/python scripts/244_build_npr_transfer_figures_v1.py",
])
npr_protocol_v1
'''.strip()

    new_cells: list[dict] = [
        markdown(
            f"""
{MARKER}

前一节提出要从生成机制方向补充证据，本节将其真正落实为可复现实验。NPR（Neighboring Pixel Relationships）不读取 EXIF、PNG 信息键或平台名称，而是先对相邻像素的上采样关系做残差变换，再由轻量卷积骨干提取低层生成痕迹。该方向来自 [CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html)，代码与权重取自[作者公开仓库](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)。

本节回答四个问题：官方权重能否直接迁移；只重训决策头能否改善跨域性能；NPR 能否补足现有模型对未知生成器的漏检；经过 JPEG、缩放和重编码后，低层痕迹还能保留多少。实验固定上游提交 `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`，主权重 SHA-256 为 `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`。
"""
        ),
        markdown(
            """
##### 6.5.11.1 数据协议与无泄漏边界

沿用 Community、Legacy、NTIRE 与 AIGenImages2026 的既有协议。训练集用于拟合新决策头，验证集用于选择样本权重、正则强度、分类阈值和复核策略；18,206 张原始图测试集与 2,236 张重编码压力集均不参与任何参数选择。四个数据区共处理 **47,167 个图像输入**，冻结骨干输出 512 维向量并做 L2 归一化。
"""
        ),
        code_table(setup_code, protocol, counter),
    ]
    counter += 1
    new_cells.extend(
        [
            markdown(
                """
##### 6.5.11.2 官方 NPR 零样本迁移不是即插即用

先在四个验证域上选择官方检查点与统一阈值，再一次性评估测试集。官方 NPR 在 SafeIMG 上表现出较高召回，但在 Community、Legacy 和 NTIRE 上存在严重域偏移：部分范围的 ROC AUC 接近或低于随机水平，真实图误报率最高接近一半。结论不是“NPR 无效”，而是原论文训练分布、输出尺度和本项目图像链路不同，不能直接把官方分数当作通用概率。
"""
            ),
            code_table("official_npr_zero_shot_metrics", official_view, counter),
        ]
    )
    counter += 1
    new_cells.extend(
        [
            markdown(
                f"""
##### 6.5.11.3 冻结骨干迁移：只学习新的透明决策边界

为区分“表征没有价值”与“原决策边界不适配”，本研究冻结官方 NPR 卷积骨干，将最后分类层替换为恒等映射，提取 512 维平均池化向量；随后训练 `StandardScaler + LogisticRegression`。候选包括统一权重、类别平衡、域与类别双平衡三种方案，以及 `C=0.001/0.01/0.1/1` 四档正则强度。每个模型只在验证集搜索 181 个阈值，共形成 **{len(head_candidates):,} 组模型与阈值候选**。

选模分数为 `最差域平衡准确率 + 0.50×平均平衡准确率 - 0.35×最高真实图误报率 - 0.10×域间标准差`。最终选择域与类别双平衡、`C={selected_head['regularization_c']}`、阈值 `{selected_head['threshold']}`。该目标明确惩罚最差域和真实图误报，不允许大数据域的高分掩盖小域失败。
"""
            ),
            code_table("npr_transfer_candidate_audit", candidate_audit, counter),
        ]
    )
    counter += 1
    new_cells.extend(
        [
            code_table("npr_transfer_cross_domain_metrics", comparison_view, counter),
            code_image(FIGURE_DIR / "npr_cross_domain_comparison.png", counter + 1),
            markdown(
                """
**结果解释。** 迁移头在 Community 测试上仍有 0.8949 的平衡准确率，但在 NTIRE 标准、NTIRE 困难和 AIGenImages2026 上分别只有 0.5701、0.5304 和 0.7236，均低于现有三域模型。概率融合在验证集上选择 NPR 权重 0.125 和阈值 0.74，虽然把两类测试域的最高真实图误报率从 0.1920 压到 0.0424，却使平均平衡准确率从 0.7733 降到 0.7381，说明它主要通过保守地少报 AI 来换取低误报，不能视为整体升级。
"""
            ),
        ]
    )
    counter += 2
    new_cells.extend(
        [
            markdown(
                """
##### 6.5.11.4 未知生成器揭示了互补价值，也否定了简单融合

SafeIMG 只有生成图，不能计算真实图误报率，因此这里只讨论召回而不宣称完整准确率。现有三域模型在 SafeIMG 上召回率为 0.3775；官方 NPR 零样本为 0.8718；重新训练的 NPR 迁移头为 0.8647。相反，验证集选择的概率融合只有 0.1671。这个反常结果说明两个模型的分数并不处于同一校准空间，线性混合会把互补信号抵消。NPR 的正确角色应是独立证据或分歧告警，而不是无条件平均概率。
"""
            ),
            code_table("npr_unknown_generator_metrics", unknown_view, counter),
            code_image(FIGURE_DIR / "npr_unknown_generator_recall.png", counter + 1, width=980),
        ]
    )
    counter += 2
    new_cells.extend(
        [
            markdown(
                """
##### 6.5.11.5 传播压力测试：低层痕迹同样会衰减

迁移头与阈值在原图测试后保持冻结。AIGenImages2026 原始文件上的平衡准确率为 0.7236；JPEG Q90 后降到 0.5680；先缩放到 75% 再以 JPEG85 编码后为 0.5564。生成图召回率从 0.7728 分别降到 0.3560 和 0.2952。这说明 NPR 比文件元数据更接近图像内容，但仍不是不可破坏的稳定水印；插值和有损编码会改写邻域像素关系。
"""
            ),
            code_table("npr_reencoding_stress_metrics", stress_view, counter),
            code_image(FIGURE_DIR / "npr_reencoding_stress.png", counter + 1, width=1000),
        ]
    )
    counter += 2
    new_cells.extend(
        [
            markdown(
                """
##### 6.5.11.6 第二意见复核通道：不自动改判，只增加可审计的复核建议

为利用 SafeIMG 上的互补信号，同时避免 NPR 直接误伤真实图，本研究设计“分歧转复核”策略：只有当现有模型判为真实、样本尚未进入原复核队列，并且 NPR 概率超过阈值时，才新增人工复核。阈值仍只在四个验证域选择，并分别限制任一验证域真实图新增复核率不超过 1%、3% 或 5%。

3% 档最终阈值为 0.8343；验证集中实际最高真实图新增复核率为 2.6%，总体挽回 3.70% 的现有模型未复核漏检。该比例在常规域不高，但在 SafeIMG 独立生成集上可将 53.96% 的剩余漏检送入复核，证明它是有价值的异常告警信号。同时，SafeIMG 的剩余自动覆盖率仅 28.82%，且缺少配对真实图，因此不能把这一结果转成线上自动定责规则。
"""
            ),
            code_table("npr_review_guard_validation_policies", policy_view, counter),
            code_table("npr_review_guard_test_metrics", review_view, counter + 1),
            code_image(FIGURE_DIR / "npr_review_guard_tradeoff.png", counter + 2),
        ]
    )
    counter += 3
    commands = "\n".join(
        [
            "# PyTorch 2.9.1 environment: extract official NPR frozen embeddings",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/241_extract_npr_embeddings_v1.py",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/241_extract_npr_embeddings_v1.py --splits stress",
            "# Project environment: train the transfer head and evaluate the review guard",
            "./.venv/bin/python scripts/242_train_npr_transfer_head_v1.py",
            "./.venv/bin/python scripts/243_evaluate_npr_transfer_review_guard_v1.py",
            "./.venv/bin/python scripts/244_build_npr_transfer_figures_v1.py",
        ]
    )
    new_cells.extend(
        [
            markdown(
                """
##### 6.5.11.7 研究结论、部署边界与后续方向

1. NPR 提供了与 CLIP 语义特征、视觉统计和文件元数据不同的低层证据，SafeIMG 结果证明这种互补性真实存在。
2. 冻结骨干后重训逻辑回归头可以修正部分阈值与数据域偏移，但不能解决所有跨域差异。
3. 简单概率融合不是多证据系统；当模型校准和域适用性不同，融合可能同时降低已知域准确率和未知域召回。
4. 当前最佳用途是“第二意见转复核”，而不是替换主模型或直接自动改判。
5. JPEG 和缩放会削弱 NPR 信号，未来应加入重编码增强，并进一步验证频域模型、扩散重建误差和来源凭证。
6. 作者仓库当前快照没有明确 `LICENSE` 文件。代码和权重仅保留为研究复现实验，未打包进公开网站，也未改变线上主模型。

因此，本轮的实质进展不是把网页数字做得更高，而是新增了一条独立技术路线，量化了它能补足什么、会在哪些条件下失效，并把可用价值收敛为有边界的治理策略。
"""
            ),
            code_stdout("print(npr_reproduction_commands)", commands, counter),
        ]
    )

    for index, cell in enumerate(new_cells):
        cell["id"] = f"npr-transfer-v1-{index:02d}"

    cells[insert_at:insert_at] = new_cells
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"updated={NOTEBOOK.relative_to(ROOT)}")
    print(f"inserted_cells={len(new_cells)}")
    print(f"selected_fusion_weight={selected_fusion['npr_head_weight']}")
    print(f"review_threshold={review_summary['selected_policy']['threshold']}")


if __name__ == "__main__":
    main()
