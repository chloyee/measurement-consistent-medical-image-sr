#!/usr/bin/env python3
"""Registered SPARX module ablation and measurement-consistency gate."""

from __future__ import annotations

import argparse
import json


VARIANTS = {
    "full": {"kd": True, "large_field": True, "learned_gate": True, "exact_projection": True},
    "no_kd": {"kd": False, "large_field": True, "learned_gate": True, "exact_projection": True},
    "no_large_field": {"kd": True, "large_field": False, "learned_gate": False, "exact_projection": True},
    "ungated": {"kd": True, "large_field": True, "learned_gate": False, "exact_projection": True},
    "no_projection": {"kd": True, "large_field": True, "learned_gate": True, "exact_projection": False},
}

PROTOCOL = {
    "backbone": "C96D14",
    "seed": 2026,
    "max_steps": 300_000,
    "stop_at_step": 120_000,
    "scheduler_total_steps": 300_000,
    "early_stopping": {"enabled": False},
    "initialization": "matched pre-adaptation checkpoint",
    "selection_partition": "LiTS val",
    "measurement_consistency_limit_hu": 1e-3,
}


def eligible(record: dict) -> bool:
    """Report consistency as a diagnostic, not a model-selection rule."""
    return float(record["data_consistency_hu"]) <= PROTOCOL["measurement_consistency_limit_hu"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", help="optional aggregate JSON list")
    args = parser.parse_args()
    payload: dict = {"variants": VARIANTS, "protocol": PROTOCOL}
    if args.records:
        with open(args.records, encoding="utf-8") as handle:
            records = json.load(handle)
        payload["records"] = [{**item, "passes_consistency_check": eligible(item)} for item in records]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
