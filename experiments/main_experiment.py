"""Train a SPARX teacher or student using patient-disjoint slice data."""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from data_utils import SliceDataset, bicubic_downsample, mix_liver_window_views
from metrics import batch_metrics
from util import build_model


def reconstruction_loss(output, target, domain, preliminary_weight, window_weight=.5):
    weights = torch.where(domain.flatten() > .5, window_weight, 1.0)
    def loss(pred):
        return ((pred-target).abs().flatten(1).mean(1)*weights).sum()/weights.sum()
    return loss(output["sr"]) + preliminary_weight*loss(output["preliminary_sr"])


def kd_loss(student, teacher, domain):
    wide = (domain.flatten() < .5).to(student.dtype)
    return ((student-teacher).abs().flatten(1).mean(1)*wide).sum()/wide.sum().clamp_min(1)


def preliminary_weight(step, cfg):
    start, end = cfg["preliminary_anneal"]
    return cfg["preliminary_weight"] * min(1., max(0., (end-step)/(end-start)))


def learning_rate(step, cfg):
    if step <= cfg["warmup_steps"]:
        return cfg["lr"]*max(step, 1)/cfg["warmup_steps"]
    progress = max(0., (step-cfg["hold_until_step"])/(cfg["max_steps"]-cfg["hold_until_step"]))
    return cfg["eta_min"] + .5*(cfg["lr"]-cfg["eta_min"])*(1+math.cos(math.pi*min(progress,1.)))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    series = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        hr = batch["hr"].to(device)
        lr = bicubic_downsample(hr, 2)
        sr = model(lr)["sr"]
        for metric, values in batch_metrics(sr, hr, lr, 2).items():
            for patient, scan, value in zip(batch["patient_id"], batch["series_id"], values.tolist()):
                series[(patient, scan)][metric].append(value)
    patients = defaultdict(lambda: defaultdict(list))
    for (patient, _), values in series.items():
        for key, items in values.items():
            patients[patient][key].append(float(np.mean(items)))
    if not patients:
        raise ValueError("Empty evaluation loader")
    return {key: float(np.mean([np.mean(v[key]) for v in patients.values()]))
            for key in next(iter(patients.values()))}


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t = cfg["train"]
    for key in ("lr", "eta_min", "weight_decay", "preliminary_weight",
                "window_probability", "window_weight", "kd_weight"):
        t[key] = float(t[key])
    if t["batch_size"] < 2:
        raise ValueError("Training batch size must be at least 2")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_set = SliceDataset(cfg["data"]["manifest"], "train", cfg["data"]["patch_size"], True)
    val_set = SliceDataset(cfg["data"]["manifest"], "val", cfg["data"].get("evaluation_patch_size"))
    if len(train_set) < t["batch_size"]:
        raise ValueError("Training partition needs at least one complete batch")
    loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["workers"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=t["workers"])
    model = build_model(cfg["model"]).to(device)
    teacher = None
    if t["kd_weight"] > 0:
        teacher_state = torch.load(t["teacher_checkpoint"], map_location="cpu", weights_only=True)
        if teacher_state["config"]["model"] != cfg["model"]:
            raise ValueError("Teacher and student must use the same architecture")
        trained_patients = set(teacher_state.get("training_patients", []))
        current_patients = {r["patient_id"] for r in train_set.rows}
        if not trained_patients or trained_patients != current_patients:
            raise ValueError("Teacher and student must use the same training patients")
        teacher = build_model(cfg["model"]).to(device).eval()
        teacher.load_state_dict(teacher_state["model"], strict=True)
        teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.pt"
    if last.exists() and not args.resume:
        raise FileExistsError("Use --resume or select a new output directory")
    start, best, stop_best, bad = 0, -math.inf, -math.inf, 0
    if args.resume:
        state = torch.load(last, map_location=device, weights_only=True)
        if state["config"] != cfg:
            raise ValueError("Resume requires the saved configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_psnr"]
        stop_best, bad = state["stop_best"], state["bad_validations"]
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    iterator = iter(loader)
    for step in range(start+1, t["max_steps"]+1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        hr = batch["hr"].to(device)
        hr, domain = mix_liver_window_views(hr, t["window_probability"], batch["lower_hu"], batch["width_hu"])
        lr = bicubic_downsample(hr, 2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, t)
        optimizer.zero_grad(set_to_none=True)
        output = model(lr, window_domain=domain)
        loss = reconstruction_loss(output, hr, domain, preliminary_weight(step,t), t["window_weight"])
        if teacher is not None:
            wide = domain.flatten() < .5
            if wide.any():
                with torch.no_grad():
                    teacher_sr = teacher(lr[wide])["sr"]
                loss = loss + t["kd_weight"] * (output["sr"][wide]-teacher_sr).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % t["log_every"] == 0:
            print(json.dumps({"step":step, "loss":float(loss.detach()), "lr":optimizer.param_groups[0]["lr"]}), flush=True)
        if step % t["validate_every"] == 0 or step == t["max_steps"]:
            metrics = evaluate(model, val_loader, device)
            score = metrics["psnr"]
            improved = score > best
            best = max(best, score)
            if score > stop_best + t["early_stopping"]["min_delta"]:
                stop_best, bad = score, 0
            elif step >= t["early_stopping"]["start_step"]:
                bad += 1
            state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(),
                     "config":cfg, "step":step, "best_psnr":best, "stop_best":stop_best,
                     "bad_validations":bad, "training_patients":sorted({r["patient_id"] for r in train_set.rows})}
            save_checkpoint(last, state)
            if improved:
                save_checkpoint(out / "best.pt", state)
            with (out / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps({"step":step, **metrics})+"\n")
            print(json.dumps({"step":step, "validation":metrics}), flush=True)
            if step >= t["early_stopping"]["start_step"] and bad >= t["early_stopping"]["patience"]:
                break


if __name__ == "__main__":
    main()
