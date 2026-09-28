#!/usr/bin/env python3
"""Extract frozen official NPR backbone embeddings for strict train/valid/test splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/239_evaluate_official_npr_v1.py"
OUT_DIR = ROOT / "outputs/features/npr_official_embeddings_v1"
TABLE_DIR = ROOT / "outputs/tables"
WEIGHTS = ROOT / "third_party/NPR-DeepfakeDetection-main/NPR.pth"


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("official_npr_eval_v1", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_train_manifest(helper: Any) -> pd.DataFrame:
    modern = pd.read_csv(helper.MODERN_MANIFEST, low_memory=False)
    modern["ntire_split"] = helper.assign_ntire_protocol(modern)
    aigen = pd.read_csv(helper.AIGEN_MANIFEST, low_memory=False)
    aigen["aigen_split"] = helper.assign_aigen_protocol(aigen)
    parts = [
        helper.normalize_rows(
            modern[
                modern["source_family"].eq("community_forensics_small")
                & modern["protocol_split"].eq("train")
            ],
            "community_train",
        ),
        helper.normalize_rows(
            modern[
                modern["source_family"].eq("legacy_cifake_genimage")
                & modern["protocol_split"].eq("train")
            ],
            "legacy_train",
        ),
        helper.normalize_rows(
            modern[
                modern["source_family"].eq("ntire2026_validation")
                & modern["ntire_split"].eq("adapt_train")
            ],
            "ntire_standard_adapt_train",
        ),
        helper.normalize_rows(aigen[aigen["aigen_split"].eq("adapt_train")], "aigen2026_adapt_train"),
    ]
    frame = pd.concat(parts, ignore_index=True).sort_values(
        ["eval_scope", "experiment_sample_id"]
    ).reset_index(drop=True)
    if frame["experiment_sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample ids in NPR training manifest")
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_split(
    helper: Any,
    frame: pd.DataFrame,
    split: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    reset: bool,
) -> dict[str, Any]:
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = split_dir / "embeddings.npy"
    status_path = split_dir / "status.npy"
    index_path = TABLE_DIR / f"npr_official_embedding_{split}_manifest_v1.csv"
    frame = frame.reset_index(drop=True).copy()
    frame.insert(0, "embedding_row", np.arange(len(frame), dtype=int))
    frame.to_csv(index_path, index=False, encoding="utf-8-sig")
    if reset:
        embeddings_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
    if embeddings_path.exists() != status_path.exists():
        raise RuntimeError(f"Incomplete embedding cache for {split}")
    dimension = 512
    if embeddings_path.exists():
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        status = np.lib.format.open_memmap(status_path, mode="r+")
        if embeddings.shape != (len(frame), dimension) or status.shape != (len(frame),):
            raise RuntimeError(f"Embedding shape changed for {split}; rerun with --reset")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype="float32",
            shape=(len(frame), dimension),
        )
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype="uint8", shape=(len(frame),))
        embeddings[:] = 0.0
        status[:] = 0
        embeddings.flush()
        status.flush()

    pending_indices = np.flatnonzero(status == 0)
    pending = frame.iloc[pending_indices].reset_index(drop=True)
    if len(pending):
        dataset = helper.ManifestDataset(pending)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        completed = int((status == 1).sum())
        with torch.inference_mode():
            for images, local_indices in loader:
                vectors = model(images.to(device)).reshape(len(images), dimension)
                vectors = torch.nn.functional.normalize(vectors, dim=1)
                original_indices = pending_indices[local_indices.numpy()]
                embeddings[original_indices] = vectors.detach().cpu().numpy().astype("float32")
                status[original_indices] = 1
                completed += len(original_indices)
                if completed % 1024 < len(original_indices) or completed == len(frame):
                    print(f"split={split} progress={completed}/{len(frame)}", flush=True)
                embeddings.flush()
                status.flush()
    if not np.all(status == 1):
        raise RuntimeError(f"Incomplete NPR embedding extraction for {split}")
    return {
        "split": split,
        "rows": len(frame),
        "dimension": dimension,
        "embeddings": str(embeddings_path.relative_to(ROOT)),
        "status": str(status_path.relative_to(ROOT)),
        "index": str(index_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test", "stress"],
        default=["train", "valid", "test"],
    )
    args = parser.parse_args()
    helper = load_helper()
    train = build_train_manifest(helper)
    valid, test_with_stress = helper.build_protocol_manifests()
    test = test_with_stress[test_with_stress["condition"].eq("original")].reset_index(drop=True)
    stress = test_with_stress[~test_with_stress["condition"].eq("original")].reset_index(drop=True)
    manifests = {"train": train, "valid": valid, "test": test, "stress": stress}
    for split, frame in manifests.items():
        frame.to_csv(
            ROOT / f"data/processed/experiment_tables/npr_embedding_{split}_manifest_v1.csv",
            index=False,
            encoding="utf-8-sig",
        )

    torch.manual_seed(42)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = helper.load_model(WEIGHTS, device)
    model.fc1 = torch.nn.Identity()
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    requested_outputs = [
        extract_split(helper, manifests[split], split, model, device, args.batch_size, args.reset)
        for split in args.splits
    ]
    requested_by_split = {item["split"]: item for item in requested_outputs}
    outputs: list[dict[str, Any]] = []
    for split, frame in manifests.items():
        if split in requested_by_split:
            outputs.append(requested_by_split[split])
            continue
        embeddings_path = OUT_DIR / split / "embeddings.npy"
        status_path = OUT_DIR / split / "status.npy"
        index_path = TABLE_DIR / f"npr_official_embedding_{split}_manifest_v1.csv"
        if not (embeddings_path.exists() and status_path.exists() and index_path.exists()):
            continue
        embeddings = np.load(embeddings_path, mmap_mode="r")
        status = np.load(status_path, mmap_mode="r")
        if embeddings.shape == (len(frame), 512) and status.shape == (len(frame),) and np.all(status == 1):
            outputs.append(
                {
                    "split": split,
                    "rows": len(frame),
                    "dimension": 512,
                    "embeddings": str(embeddings_path.relative_to(ROOT)),
                    "status": str(status_path.relative_to(ROOT)),
                    "index": str(index_path.relative_to(ROOT)),
                }
            )
    metadata = {
        "status": "completed",
        "representation": "official_npr_frozen_backbone_avgpool_l2_normalized",
        "dimension": 512,
        "weight_path": str(WEIGHTS.relative_to(ROOT)),
        "weight_sha256": sha256(WEIGHTS),
        "preprocessing": "RGB, resize 256x256, center crop 224, ImageNet normalization",
        "device": str(device),
        "requested_splits": args.splits,
        "outputs": outputs,
        "license_boundary": "Research only; upstream snapshot has no explicit license file.",
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
