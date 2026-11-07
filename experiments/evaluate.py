"""Evaluate a frozen SPARX checkpoint or reconstruct a single LR slice."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from data_utils import SliceDataset, load_slice
from util import build_model
from main_experiment import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--input", help="Normalized LR .npy or .npz slice")
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=["val", "test"], default="test")
    parser.add_argument("--window-domain", choices=["wide", "liver"], default="wide")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"]["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.runtime_window_domain = args.window_domain == "liver"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.input:
        image, _, _ = load_slice(args.input)
        with torch.inference_mode():
            sr = model(image.unsqueeze(0).to(device))["sr"][0,0].cpu().numpy()
        with output.open("wb") as handle:
            np.save(handle, sr)
    else:
        dataset = SliceDataset(args.manifest, args.partition)
        patients = {r["patient_id"] for r in dataset.rows}
        if patients & set(checkpoint.get("training_patients", [])):
            raise ValueError("Evaluation patients overlap the checkpoint training patients")
        result = {"partition":args.partition, "checkpoint_step":checkpoint["step"],
                  "patients":len(patients), "slices":len(dataset),
                  "aggregation":"slice_then_series_then_patient_mean",
                  "mae_units":"normalized intensity", "psnr_peak":1.0,
                  **evaluate(model, DataLoader(dataset,batch_size=1), device)}
        output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
