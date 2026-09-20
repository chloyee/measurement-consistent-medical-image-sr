#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import LitsSlices, bicubic_downsample
from metrics import batch_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("val", "test"), default="test")
    parser.add_argument("--manifest", default="data_manifests/lits17_slices.csv")
    parser.add_argument("--output", default="experiments/bicubic_d1_x2")
    parser.add_argument("--slices-per-patient", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LitsSlices(args.manifest, args.partition, 256, False, val_slices_per_patient=args.slices_per_patient)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    patient_values = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for batch in loader:
            hr = batch["hr"].to(device)
            lr = bicubic_downsample(hr, 2)
            sr = torch.nn.functional.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            metrics = batch_metrics(sr, hr, lr, 2)
            patient = batch["patient_id"][0]
            for name, value in metrics.items(): patient_values[patient][name].append(float(value.item()))
    patients = {patient: {name: float(np.mean(values)) for name, values in metrics.items()}
                for patient, metrics in patient_values.items()}
    summary = {name: float(np.mean([metrics[name] for metrics in patients.values()]))
               for name in next(iter(patients.values()))}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.partition}_patient_metrics.json").write_text(json.dumps(patients, indent=2) + "\n")
    (out / f"{args.partition}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({"status": "completed", "partition": args.partition, **summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
