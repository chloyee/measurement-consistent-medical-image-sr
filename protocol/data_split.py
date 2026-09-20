#!/usr/bin/env python3
"""Build a deterministic patient split and slice index for LiTS17."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 2026


def volume_id(path: Path) -> int:
    return int(re.search(r"(\d+)$", path.stem).group(1))


def main() -> None:
    source = Path("dataset/LiST17/seg")
    prepared = Path("data/LITS17/train")
    manifest_dir = Path("data_manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    slice_rows = []
    for mask_path in sorted(source.glob("segmentation-*.nii"), key=volume_id):
        idx = volume_id(mask_path)
        patient = f"volume-{idx}"
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
        liver_by_slice = np.count_nonzero(mask > 0, axis=(0, 1))
        tumor_by_slice = np.count_nonzero(mask == 2, axis=(0, 1))
        tumor_voxels = int(tumor_by_slice.sum())
        liver_voxels = int(liver_by_slice.sum())
        stats.append({
            "patient_id": patient,
            "tumor_voxels": tumor_voxels,
            "liver_voxels": liver_voxels,
            "tumor_fraction": tumor_voxels / max(liver_voxels, 1),
            "slice_count": int(mask.shape[2]),
        })
        for z in range(mask.shape[2]):
            slice_rows.append({
                "patient_id": patient,
                "slice_index": z,
                "image_path": (prepared / "images" / patient / f"slice_{z:04d}.png").as_posix(),
                "mask_path": (prepared / "masks" / patient / f"slice_{z:04d}.png").as_posix(),
                "liver_pixels": int(liver_by_slice[z]),
                "tumor_pixels": int(tumor_by_slice[z]),
            })

    positive = np.array([x["tumor_voxels"] for x in stats if x["tumor_voxels"] > 0])
    quartiles = np.quantile(positive, [0.25, 0.5, 0.75])
    strata = []
    for row in stats:
        if row["tumor_voxels"] == 0:
            strata.append("no_tumor")
        else:
            strata.append(f"tumor_q{int(np.digitize(row['tumor_voxels'], quartiles)) + 1}")
    patients = [row["patient_id"] for row in stats]
    train, held, _, held_strata = train_test_split(
        patients, strata, train_size=91, random_state=SEED, stratify=strata
    )
    val, test = train_test_split(
        held, train_size=20, random_state=SEED + 1, stratify=held_strata
    )
    split = {"seed": SEED, "train": sorted(train), "val": sorted(val), "test": sorted(test)}
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    split["sha256"] = hashlib.sha256(canonical).hexdigest()
    (manifest_dir / "split_v1.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    partition = {patient: name for name in ("train", "val", "test") for patient in split[name]}
    for row in stats:
        row["partition"] = partition[row["patient_id"]]
    with (manifest_dir / "lits17_patient_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["patient_id", "partition", "tumor_voxels", "liver_voxels", "tumor_fraction", "slice_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sorted(stats, key=lambda x: x["patient_id"]))
    for row in slice_rows:
        row["partition"] = partition[row["patient_id"]]
    with (manifest_dir / "lits17_slices.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["patient_id", "partition", "slice_index", "image_path", "mask_path", "liver_pixels", "tumor_pixels"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(slice_rows)
    print(json.dumps({k: len(split[k]) for k in ("train", "val", "test")} | {"sha256": split["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
