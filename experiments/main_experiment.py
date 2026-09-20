#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data_utils import LitsSlices, bicubic_downsample, mix_liver_window_views
from metrics import batch_metrics
from util import build_model


class EarlyStopper:
    def __init__(self, config: dict):
        self.enabled = bool(config["enabled"])
        self.monitor = str(config["monitor"])
        self.mode = str(config["mode"])
        self.patience = int(config["patience_validations"])
        self.min_delta = float(config["min_delta"])
        self.min_steps = int(config["min_steps"])
        if self.mode not in {"max", "min"}:
            raise ValueError(f"early_stopping.mode must be max or min, got {self.mode}")
        if self.patience < 1:
            raise ValueError("early_stopping.patience_validations must be >= 1")
        self.best: float | None = None
        self.bad_validations = 0

    def update(self, metrics: dict[str, float], step: int) -> bool:
        if self.monitor not in metrics:
            raise KeyError(f"Early-stopping metric is unavailable: {self.monitor}")
        current = float(metrics[self.monitor])
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta
        if improved:
            self.best = current
            self.bad_validations = 0
        elif step >= self.min_steps:
            self.bad_validations += 1
        return self.enabled and step >= self.min_steps and self.bad_validations >= self.patience

    def state_dict(self) -> dict[str, float | int | None]:
        return {"best": self.best, "bad_validations": self.bad_validations}

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state["best"]) if state.get("best") is not None else None
        self.bad_validations = int(state.get("bad_validations", 0))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def super_resolution_loss(
    output: dict[str, torch.Tensor], hr: torch.Tensor, preliminary_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if preliminary_weight < 0:
        raise ValueError("preliminary_loss_weight must be non-negative")
    def weighted_l1(prediction: torch.Tensor) -> torch.Tensor:
        per_sample = (prediction - hr).abs().flatten(1).mean(1)
        if sample_weight is None:
            return per_sample.mean()
        weights = sample_weight.reshape(sample_weight.shape[0]).to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    projected = weighted_l1(output["sr"])
    components = {"projected_l1": projected}
    total = projected
    if preliminary_weight:
        if "preliminary_sr" not in output:
            raise KeyError("preliminary_sr is required when preliminary_loss_weight > 0")
        preliminary = weighted_l1(output["preliminary_sr"])
        components["preliminary_l1"] = preliminary
        total = total + preliminary_weight * preliminary
    return total, components


def preliminary_loss_weight(train_config: dict, step: int) -> float:
    """Return the optional linearly annealed preliminary-image loss weight."""
    initial = float(train_config.get("preliminary_loss_weight", 0.0))
    anneal = train_config.get("preliminary_loss_anneal")
    if not anneal:
        return initial
    start = int(anneal["start_step"])
    end = int(anneal["end_step"])
    final = float(anneal.get("final_weight", 0.0))
    if not 0 <= final <= initial:
        raise ValueError("preliminary loss final_weight must be in [0, initial_weight]")
    if not 0 <= start < end:
        raise ValueError("preliminary loss anneal requires 0 <= start_step < end_step")
    if step <= start:
        return initial
    if step >= end:
        return final
    progress = (step - start) / (end - start)
    return initial + progress * (final - initial)


@torch.inference_mode()
def validate(model, loader, device, scale):
    model.eval()
    totals = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        hr = batch["hr"].to(device, non_blocking=True)
        lr = bicubic_downsample(hr, scale)
        output = model(lr)["sr"]
        for name, tensor in batch_metrics(output, hr, lr, scale).items():
            for patient, value in zip(batch["patient_id"], tensor.cpu().tolist()):
                totals[patient][name].append(value)
    if not totals:
        raise ValueError("Evaluation partition has no slices")
    names = next(iter(totals.values()))
    return {name: float(np.mean([np.mean(v[name]) for v in totals.values()])) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--partition", choices=("val", "test"), default="test")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.evaluate_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required with --evaluate-only")
        import hashlib
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg["model"]).to(device)
        checkpoint_path = Path(args.checkpoint)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        dataset = LitsSlices(cfg["data"]["manifest"], args.partition,
                             cfg["data"]["val_patch_size"], False, cfg["seed"],
                             cfg["data"]["val_slices_per_patient"])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result = {"partition": args.partition, "checkpoint_step": state.get("step"),
                  "checkpoint_sha256": digest.hexdigest(), "slices": len(dataset),
                  "patients": len({row["patient_id"] for row in dataset.rows}),
                  "aggregation": "slice_mean_then_patient_mean",
                  **validate(model, loader, device, int(cfg["model"].get("scale", 2)))}
        atomic_json(ROOT / "reports" / cfg["experiment_id"] / (args.partition + ".json"), result)
        print(json.dumps(result), flush=True)
        return 0
    if args.smoke:
        cfg["train"]["stop_at_step"] = 20
        cfg["train"]["max_steps"] = 20; cfg["train"]["validate_every"] = 10
        cfg["train"]["batch_size"] = min(
            int(cfg["train"].get("batch_size", 2)), 2
        )
        cfg["data"]["val_slices_per_patient"] = min(
            int(cfg["data"].get("val_slices_per_patient", 8)), 1
        )
        cfg["model"]["use_raft"] = False
        if int(cfg["train"].get("warmup_steps", 0)):
            cfg["train"]["warmup_steps"] = 5
        if "hold_until_step" in cfg["train"]:
            cfg["train"]["hold_until_step"] = 10
        if cfg["train"].get("scheduler", {}).get("type") == "multistep":
            cfg["train"]["scheduler"]["milestones"] = [10, 15]
        cfg["experiment_id"] += "_smoke"
    max_steps = int(cfg["train"]["max_steps"])
    stop_at_step = int(cfg["train"].get("stop_at_step", max_steps))
    if not 0 < stop_at_step <= max_steps:
        raise ValueError("stop_at_step must be in (0, max_steps]")
    early_config = cfg["train"].setdefault("early_stopping", {})
    early_config.setdefault("enabled", True)
    early_config.setdefault("monitor", "psnr")
    early_config.setdefault("mode", "max")
    early_config.setdefault("patience_validations", 8)
    early_config.setdefault("min_delta", 0.002)
    early_config.setdefault("min_steps", 160000)
    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("CUDA is required for a non-smoke experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    out = ROOT / "experiments" / cfg["experiment_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    atomic_json(out / "status.json", {"status": "running", "step": 0})

    train_set = LitsSlices(cfg["data"]["manifest"], "train", cfg["data"]["patch_size"], True, seed)
    val_set = LitsSlices(cfg["data"]["manifest"], "val", cfg["data"].get("val_patch_size"), False, seed,
                         cfg["data"].get("val_slices_per_patient", 8))
    workers = 0 if args.smoke else cfg["train"].get("workers", 8)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=workers,
                              pin_memory=device.type == "cuda", drop_last=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    model = build_model(cfg["model"]).to(device)
    initialization = cfg.get("initialization")
    if initialization:
        if "checkpoint_path" in initialization:
            source_checkpoint_path = ROOT / initialization["checkpoint_path"]
        else:
            source = ROOT / "experiments" / initialization["experiment"]
            source_checkpoint_path = source / initialization.get(
                "checkpoint", "best_val_psnr.pt"
            )
        source_checkpoint = torch.load(
            source_checkpoint_path,
            map_location="cpu", weights_only=False,
        )
        incompatible = model.load_state_dict(
            source_checkpoint["model"], strict=False
        )
        allowed_missing = {"liver_shallow.weight", "liver_shallow.bias"}
        allowed_unexpected_prefixes = tuple(
            str(prefix)
            for prefix in initialization.get("allow_unexpected_prefixes", [])
        )
        unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.startswith(allowed_unexpected_prefixes)
        ]
        if set(incompatible.missing_keys) - allowed_missing or unexpected:
            raise RuntimeError(
                "unexpected initialization mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={unexpected}"
            )
        if initialization.get("copy_wide_to_liver_stem", False):
            if not hasattr(model, "initialize_liver_stem_from_wide"):
                raise TypeError("model cannot copy a wide stem into a liver stem")
            model.initialize_liver_stem_from_wide()

    teacher = None
    distillation = cfg["train"].get("distillation", {})
    if distillation.get("enabled", False):
        if "config_path" in distillation:
            teacher_config_path = ROOT / distillation["config_path"]
            teacher_checkpoint_path = ROOT / distillation["checkpoint_path"]
        else:
            teacher_directory = ROOT / "experiments" / distillation["teacher_experiment"]
            teacher_config_path = teacher_directory / "resolved_config.yaml"
            teacher_checkpoint_path = teacher_directory / distillation.get(
                "checkpoint", "best_val_psnr.pt"
            )
        teacher_config = yaml.safe_load(teacher_config_path.read_text(encoding="utf-8"))
        teacher = build_model(teacher_config["model"]).to(device).eval()
        teacher_checkpoint = torch.load(
            teacher_checkpoint_path,
            map_location="cpu", weights_only=False,
        )
        teacher.load_state_dict(teacher_checkpoint["model"], strict=True)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler_config = cfg["train"].get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "warmup_hold_cosine")
    scheduler_total_steps = int(scheduler_config.get("total_steps", max_steps))
    if scheduler_total_steps < max_steps:
        raise ValueError(
            "scheduler total_steps must be at least max_steps, got "
            f"{scheduler_total_steps} < {max_steps}"
        )
    warmup_steps = int(cfg["train"].get("warmup_steps", 0))
    hold_until_step = int(cfg["train"].get("hold_until_step", warmup_steps))
    eta_min = float(cfg["train"].get("eta_min", 1e-7))
    if scheduler_type == "multistep":
        milestones = [int(step) for step in scheduler_config["milestones"]]
        if not milestones or milestones != sorted(set(milestones)):
            raise ValueError("multistep milestones must be sorted and unique")
        if milestones[-1] >= max_steps:
            raise ValueError("multistep milestones must be smaller than max_steps")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones,
            gamma=float(scheduler_config.get("gamma", 0.5)),
        )
    elif scheduler_type != "warmup_hold_cosine":
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    elif warmup_steps:
        if not 0 < warmup_steps < max_steps:
            raise ValueError(f"warmup_steps must be in (0, max_steps), got {warmup_steps}")
        if not warmup_steps <= hold_until_step < max_steps:
            raise ValueError(
                "hold_until_step must be between warmup_steps and max_steps, "
                f"got {hold_until_step}"
            )
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=float(cfg["train"].get("warmup_start_factor", 1e-3)),
                end_factor=1.0, total_iters=warmup_steps,
            )
        ]
        milestones = [warmup_steps]
        if hold_until_step > warmup_steps:
            schedulers.append(
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0,
                    total_iters=hold_until_step - warmup_steps,
                )
            )
            milestones.append(hold_until_step)
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_total_steps - hold_until_step,
                eta_min=eta_min,
            )
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_total_steps, eta_min=eta_min
        )
    early_stopper = EarlyStopper(early_config)
    checkpoint = out / "last.pt"; start = 0; best = -math.inf
    steps_per_epoch = len(train_loader)
    started_at = datetime.now().astimezone().isoformat()
    process_wall_start = time.perf_counter()
    active_train_seconds = 0.0
    timed_steps = 0
    if checkpoint.exists() and not args.smoke:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start, best = state["step"], state["best_psnr"]
        if (
            "early_stopping" in state
            and not cfg["train"].get("reset_early_stopping_on_resume", False)
        ):
            early_stopper.load_state_dict(state["early_stopping"])
        timing_state = state.get("timing", {})
        started_at = str(timing_state.get("started_at", started_at))
        active_train_seconds = float(timing_state.get("active_train_seconds", 0.0))
        timed_steps = int(timing_state.get("timed_steps", 0))
    iterator = iter(train_loader); running = 0.0
    running_components: defaultdict[str, float] = defaultdict(float)
    stopped_early = False
    last_step = start
    try:
        for step in range(start + 1, stop_at_step + 1):
            last_step = step
            train_step_start = time.perf_counter()
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train_loader); batch = next(iterator)
            hr = batch["hr"].to(device, non_blocking=True)
            window_mix_fraction = hr.new_zeros(())
            window_domain = hr.new_zeros((hr.shape[0], 1, 1, 1))
            window_mix_probability = float(
                cfg["data"].get("window_mix_probability", 0.0)
            )
            if window_mix_probability:
                hr, window_domain = mix_liver_window_views(
                    hr, window_mix_probability
                )
                window_mix_fraction = window_domain.mean()
            lr = bicubic_downsample(hr, cfg["model"]["scale"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                model_kwargs = (
                    {"window_domain": window_domain}
                    if cfg["model"].get("dual_window_stem", False) else {}
                )
                output = model(lr, **model_kwargs)
                current_preliminary_weight = preliminary_loss_weight(
                    cfg["train"], step
                )
                window_loss_weight = float(
                    cfg["train"].get("window_loss_weight", 1.0)
                )
                sample_weight = torch.where(
                    window_domain > 0.5,
                    window_domain.new_full((), window_loss_weight),
                    window_domain.new_ones(()),
                ) if window_mix_probability else None
                loss, loss_components = super_resolution_loss(
                    output, hr, current_preliminary_weight, sample_weight
                )
                if teacher is not None:
                    with torch.no_grad():
                        teacher_sr = teacher(lr)["sr"]
                    raw_weight = (1.0 - window_domain).reshape(hr.shape[0])
                    per_sample_kd = (
                        output["sr"] - teacher_sr
                    ).abs().flatten(1).mean(1)
                    kd = (
                        (per_sample_kd * raw_weight).sum()
                        / raw_weight.sum().clamp_min(1.0)
                    )
                    kd_weight = float(distillation.get("weight", 0.1))
                    loss = loss + kd_weight * kd
                    loss_components["raw_teacher_l1"] = kd
                if window_mix_probability:
                    loss_components["window_mix_fraction"] = window_mix_fraction
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step(); running += float(loss)
            for name, value in loss_components.items():
                running_components[name] += float(value)
            active_train_seconds += time.perf_counter() - train_step_start
            timed_steps += 1
            if step % cfg["train"].get("log_every", 100) == 0:
                seconds_per_step = active_train_seconds / max(timed_steps, 1)
                log_every = cfg["train"].get("log_every", 100)
                print(json.dumps({
                    "step": step,
                    "equivalent_epoch": step / steps_per_epoch,
                    "loss": running / log_every,
                    "preliminary_loss_weight": current_preliminary_weight,
                    **{
                        name: value / log_every
                        for name, value in running_components.items()
                    },
                    "lr": scheduler.get_last_lr()[0],
                    "active_train_seconds": active_train_seconds,
                    "timed_steps": timed_steps,
                    "seconds_per_train_step": seconds_per_step,
                    "seconds_per_equivalent_epoch": seconds_per_step * steps_per_epoch,
                    "process_wall_seconds": time.perf_counter() - process_wall_start,
                }), flush=True)
                running = 0.0
                running_components.clear()
            if step % cfg["train"]["validate_every"] == 0 or step == stop_at_step:
                metrics = validate(model, val_loader, device, cfg["model"]["scale"])
                if not all(math.isfinite(x) for x in metrics.values()): raise RuntimeError(f"Non-finite metrics: {metrics}")
                record = {"step": step, **metrics}
                with (out / "validation.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
                should_stop = early_stopper.update(metrics, step)
                improved_psnr = metrics["psnr"] > best
                best = max(best, metrics["psnr"])
                timing = {
                    "started_at": started_at,
                    "steps_per_equivalent_epoch": steps_per_epoch,
                    "equivalent_epoch": step / steps_per_epoch,
                    "active_train_seconds": active_train_seconds,
                    "timed_steps": timed_steps,
                    "seconds_per_train_step": active_train_seconds / max(timed_steps, 1),
                    "seconds_per_equivalent_epoch": active_train_seconds / max(timed_steps, 1) * steps_per_epoch,
                    "process_wall_seconds": time.perf_counter() - process_wall_start,
                }
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict(), "step": step, "best_psnr": best, "config": cfg,
                         "early_stopping": early_stopper.state_dict(), "timing": timing}
                torch.save(state, checkpoint)
                if improved_psnr:
                    torch.save(state, out / "best_val_psnr.pt")
                early_status = {
                    **early_stopper.state_dict(),
                    "enabled": early_stopper.enabled,
                    "monitor": early_stopper.monitor,
                    "patience_validations": early_stopper.patience,
                }
                atomic_json(out / "status.json", {
                    "status": "running", "step": step, "best_psnr": best,
                    "latest": metrics, "early_stopping": early_status, "timing": timing,
                })
                print(json.dumps(record), flush=True); model.train()
                decision_gate = cfg["train"].get("synchronized_decision_gate")
                if decision_gate and step >= int(decision_gate["step"]):
                    reference_path = (
                        ROOT / "experiments" / decision_gate["reference_experiment"]
                        / "validation.jsonl"
                    )
                    reference_rows = [
                        json.loads(line)
                        for line in reference_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    reference = next(
                        (item for item in reference_rows if int(item["step"]) == step),
                        None,
                    )
                    if reference is None:
                        raise RuntimeError(
                            f"decision-gate reference has no validation at step {step}: "
                            f"{reference_path}"
                        )
                    metric = str(decision_gate.get("metric", "psnr"))
                    delta = float(metrics[metric]) - float(reference[metric])
                    minimum_delta = float(decision_gate["minimum_delta"])
                    print(json.dumps({
                        "event": "synchronized_decision_gate", "step": step,
                        "metric": metric, "candidate": metrics[metric],
                        "reference": reference[metric], "delta": delta,
                        "minimum_delta": minimum_delta,
                        "passed": delta >= minimum_delta,
                    }), flush=True)
                    if delta < minimum_delta:
                        stopped_early = True
                        break
                    # This screen has a single pre-registered decision point.
                    cfg["train"].pop("synchronized_decision_gate")
                if should_stop:
                    stopped_early = True
                    print(json.dumps({
                        "event": "early_stopping", "step": step,
                        "monitor": early_stopper.monitor, "best": early_stopper.best,
                        "bad_validations": early_stopper.bad_validations,
                    }), flush=True)
                    break
        atomic_json(out / "status.json", {
            "status": "early_stopped" if stopped_early else "completed",
            "step": last_step, "best_psnr": best,
            "timing": {
                "started_at": started_at,
                "finished_at": datetime.now().astimezone().isoformat(),
                "steps_per_equivalent_epoch": steps_per_epoch,
                "equivalent_epoch": last_step / steps_per_epoch,
                "active_train_seconds": active_train_seconds,
                "timed_steps": timed_steps,
                "seconds_per_train_step": active_train_seconds / max(timed_steps, 1),
                "seconds_per_equivalent_epoch": active_train_seconds / max(timed_steps, 1) * steps_per_epoch,
                "process_wall_seconds": time.perf_counter() - process_wall_start,
            },
            "early_stopping": {
                **early_stopper.state_dict(),
                "enabled": early_stopper.enabled,
                "monitor": early_stopper.monitor,
                "patience_validations": early_stopper.patience,
            },
        })
    except BaseException as error:
        atomic_json(out / "status.json", {
            "status": "failed", "step": step if "step" in locals() else start,
            "error": repr(error),
            "timing": {
                "started_at": started_at,
                "failed_at": datetime.now().astimezone().isoformat(),
                "steps_per_equivalent_epoch": steps_per_epoch,
                "active_train_seconds": active_train_seconds,
                "timed_steps": timed_steps,
                "process_wall_seconds": time.perf_counter() - process_wall_start,
            },
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
