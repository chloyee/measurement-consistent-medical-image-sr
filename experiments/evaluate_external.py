#!/usr/bin/env python3
"""Evaluate a frozen internal or official SR checkpoint on one HKU domain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ctsr.data import bicubic_downsample
from ctsr.external_data import HKUExternalSlices
from ctsr.metrics import batch_metrics
from ctsr.models import build_model
TERMINAL = {"completed", "early_stopped"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_internal(config_path: str, checkpoint_path: str) -> nn.Module:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    model = build_model(config["model"])
    checkpoint = torch.load(
        ROOT / checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def hierarchical_summary(values: dict[str, dict[str, list[float]]], metadata: dict[str, dict]) -> dict:
    series_means = {
        series: {metric: float(np.mean(items)) for metric, items in metrics.items()}
        for series, metrics in values.items()
    }
    patient_series: dict[str, list[str]] = defaultdict(list)
    for series, item in metadata.items():
        patient_series[item["patient_id"]].append(series)
    patient_means = {
        patient: {
            metric: float(np.mean([series_means[series][metric] for series in series_ids]))
            for metric in next(iter(series_means.values()))
        }
        for patient, series_ids in patient_series.items()
    }
    summary = {
        metric: float(np.mean([metrics[metric] for metrics in patient_means.values()]))
        for metric in next(iter(patient_means.values()))
    }
    return {"summary": summary, "patient_metrics": patient_means, "series_metrics": series_means}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--loader", choices=("bicubic", "internal"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--domain", choices=("contrast", "noncontrast"), required=True)
    parser.add_argument("--slices-per-series", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output = ROOT / "reports" / "external_hku" / args.method_id / f"{args.domain}.json"
    if output.exists() and json.loads(output.read_text(encoding="utf-8")).get("status") == "completed":
        print(f"[skip] {args.method_id}/{args.domain} already completed", flush=True)
        return 0
    if args.loader == "bicubic":
        model = None
    else:
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint are required")
        model = load_internal(args.config, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model = model.to(device).eval()
        if hasattr(model, "set_input_window"):
            # Recovered DICOM-to-PNG provenance: liver window C=40/W=120,
            # uint8 quantization. This affects only the optional router view.
            model.set_input_window(40.0, 120.0, 120.0 / 255.0)

    manifest = ROOT / "data" / "manifests" / f"hku_{args.domain}_slices.csv"
    dataset = HKUExternalSlices(str(manifest), args.slices_per_series, args.crop_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    status_path = output.with_name(f"{args.domain}_status.json")
    atomic_json(status_path, {"status": "running", "step": 0, "total": len(dataset)})
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metadata: dict[str, dict] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            hr = batch["hr"].to(device, non_blocking=True)
            lr = bicubic_downsample(hr, 2)
            if model is None:
                sr = torch.nn.functional.interpolate(
                    lr, scale_factor=2, mode="bicubic", align_corners=False, antialias=True
                ).clamp(0, 1)
            else:
                prediction = model(lr)
                sr = prediction["sr"] if isinstance(prediction, dict) else prediction
            metrics = batch_metrics(sr, hr, lr, 2)
            metrics["mae_intensity_8bit"] = metrics.pop("mae_hu") * (255.0 / 3024.0)
            metrics["data_consistency_8bit"] = metrics.pop("data_consistency_hu") * (255.0 / 3024.0)
            metrics["mae_hu_window"] = metrics["mae_intensity_8bit"] * (120.0 / 255.0)
            metrics["data_consistency_hu_window"] = (
                metrics["data_consistency_8bit"] * (120.0 / 255.0)
            )
            series = batch["series_id"][0]
            metadata[series] = {"patient_id": batch["patient_id"][0], "disease": batch["disease"][0]}
            for name, tensor in metrics.items():
                values[series][name].append(float(tensor.item()))
            if index == 1 or index % 100 == 0 or index == len(dataset):
                atomic_json(status_path, {"status": "running", "step": index, "total": len(dataset),
                                          "wall_seconds": time.perf_counter() - started})

    aggregate = hierarchical_summary(values, metadata)
    payload = {
        "status": "completed",
        "method_id": args.method_id,
        "domain": f"HKU-{args.domain}",
        "protocol": "D1-x2 synthetic LR; central 256 crop; 8 deterministic slices/series; series then patient equal-weight aggregation",
        "intensity_semantics": (
            "uint8 liver-window domain (center=40 HU, width=120 HU); reconstructed "
            "window-domain HU is censored at [-20, 100] HU"
        ),
        "slices": len(dataset),
        "series": len(aggregate["series_metrics"]),
        "patients": len(aggregate["patient_metrics"]),
        "wall_seconds": time.perf_counter() - started,
        **aggregate,
    }
    atomic_json(output, payload)
    atomic_json(status_path, {"status": "completed", "step": len(dataset), "total": len(dataset),
                              "summary": payload["summary"], "wall_seconds": payload["wall_seconds"]})
    print(json.dumps(payload["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
