#!/usr/bin/env python3
"""Registered SPARX capacity ablation and its pre-declared selection rule.

This review script records the experimental matrix without embedding private
checkpoint paths or patient manifests. Every row uses the same LiTS patient
split, 120k adaptation budget, WindowMix/KD losses, and evaluation protocol.
"""

from __future__ import annotations

import argparse
import json


ANCHORS = (
    {"id": "C64D8", "channels": 64, "depth": 8, "parameters": 787_000},
    {"id": "C64D14", "channels": 64, "depth": 14, "parameters": 1_231_000},
    {"id": "C64D18", "channels": 64, "depth": 18, "parameters": 1_526_000},
    {"id": "C96D14", "channels": 96, "depth": 14, "parameters": 2_759_244},
    {"id": "C128D14", "channels": 128, "depth": 14, "parameters": 4_894_000},
)

COMMON_PROTOCOL = {
    "seed": 2026,
    "scale": 2,
    "max_steps": 120_000,
    "validation_interval": 5_000,
    "patch_size": 256,
    "batch_size": 16,
    "window_mix_probability": 0.30,
    "kd_weight": 0.20,
    "preliminary_loss": {"initial": 0.10, "anneal": [80_000, 120_000]},
    "early_stopping": {"min_steps": 80_000, "patience": 8, "min_delta_db": 0.002},
}


def select(records: list[dict]) -> dict:
    """Select by LiTS accuracy subject to external robustness and cost."""
    required = {item["id"] for item in ANCHORS}
    if {item["id"] for item in records} != required:
        raise ValueError("records must contain exactly the five registered anchors")
    # Reject a candidate if both HKU domains regress from the preceding smaller
    # anchor; among the remaining candidates choose maximum LiTS PSNR, breaking
    # ties toward fewer parameters.
    eligible = []
    ordered = sorted(records, key=lambda item: item["parameters"])
    for index, item in enumerate(ordered):
        if index and item["nc_psnr"] < ordered[index - 1]["nc_psnr"] and item["ce_psnr"] < ordered[index - 1]["ce_psnr"]:
            continue
        eligible.append(item)
    return max(eligible, key=lambda item: (item["lits_psnr"], -item["parameters"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-protocol", action="store_true")
    parser.add_argument("--records", help="optional aggregate JSON list")
    args = parser.parse_args()
    print(json.dumps({"anchors": ANCHORS, "common": COMMON_PROTOCOL}, indent=2))
    if args.records:
        with open(args.records, encoding="utf-8") as handle:
            print(json.dumps({"selected": select(json.load(handle))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

