#!/usr/bin/env python3
"""Evaluate an official RGB SR checkpoint on the unified grayscale LiTS validation set."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ctsr.data import LitsSlices, bicubic_downsample
from ctsr.metrics import batch_metrics


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GrayscaleViaRGB(nn.Module):
    """Apply an official RGB model to CT by repeat-to-RGB and mean-to-gray."""

    def __init__(self, network: nn.Module, input_range: float = 1.0):
        super().__init__()
        self.network = network
        self.input_range = float(input_range)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.network(image.repeat(1, 3, 1, 1) * self.input_range)
        return output.mean(dim=1, keepdim=True) / self.input_range


def build_network(config: dict) -> nn.Module:
    variant = config["variant"]
    if variant == "swinir_official":
        sys.path.insert(0, str(ROOT / "third_party" / "SwinIR"))
        SwinIR = importlib.import_module("models.network_swinir").SwinIR
        return SwinIR(
            upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler="pixelshuffle", resi_connection="1conv",
        )
    if variant == "mambairv2_official":
        sys.path.insert(0, str(ROOT / "third_party" / "MambaIRv2"))
        MambaIRv2 = importlib.import_module(
            "basicsr.archs.mambairv2_arch"
        ).MambaIRv2
        return MambaIRv2(
            upscale=2, in_chans=3, img_size=64, img_range=1.0,
            embed_dim=174, d_state=16, depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6], window_size=16,
            inner_rank=64, num_tokens=128, convffn_kernel_size=5,
            mlp_ratio=2.0, upsampler="pixelshuffle", resi_connection="1conv",
        )
    if variant == "omnisr_official":
        sys.path.insert(0, str(ROOT / "third_party" / "OmniSR"))
        OmniSR = importlib.import_module("components.OmniSR").OmniSR
        return OmniSR(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            upsampling=2, kernel_size=17, res_num=5, block_num=1,
            bias=True, block_script_name="OSA", block_class_name="OSA_Block",
            window_size=8, pe=True, ffn_bias=True,
        )
    if variant == "dat_s_official":
        sys.path.insert(0, str(ROOT / "third_party" / "DAT"))
        DAT = importlib.import_module("basicsr.archs.dat_arch").DAT
        return DAT(
            upscale=2, in_chans=3, img_size=64, img_range=1.0,
            split_size=[8, 16], depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            expansion_factor=2, resi_connection="1conv",
        )
    if variant == "edsr_baseline_official":
        sys.path.insert(0, str(ROOT / "third_party" / "EDSR-PyTorch" / "src"))
        EDSR = importlib.import_module("model.edsr").EDSR
        return EDSR(SimpleNamespace(
            n_resblocks=16, n_feats=64, scale=[2], rgb_range=255,
            n_colors=3, res_scale=1.0,
        ))
    if variant == "edsr_official":
        sys.path.insert(0, str(ROOT / "third_party" / "EDSR-PyTorch" / "src"))
        EDSR = importlib.import_module("model.edsr").EDSR
        return EDSR(SimpleNamespace(
            n_resblocks=32, n_feats=256, scale=[2], rgb_range=255,
            n_colors=3, res_scale=0.1,
        ))
    if variant == "rcan_official":
        sys.path.insert(
            0, str(ROOT / "third_party" / "RCAN" / "RCAN_TestCode" / "code")
        )
        RCAN = importlib.import_module("model.rcan").RCAN
        return RCAN(SimpleNamespace(
            n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16,
            scale=[2], rgb_range=255, n_colors=3, res_scale=1.0,
            data_train="DIV2K",
        ))
    if variant in {"pft_sr_class_official", "pft_sr_light_official"}:
        source = ROOT / "third_party" / "PFT-SR"
        sys.path.insert(0, str(source))
        PFT = importlib.import_module("basicsr.archs.pft_arch").PFT
        return PFT(
            upscale=2, in_chans=3, img_size=64,
            embed_dim=int(config["embed_dim"]),
            depths=tuple(config["depths"]),
            num_heads=int(config["num_heads"]),
            num_topk=list(config["num_topk"]),
            window_size=32, convffn_kernel_size=7,
            img_range=1.0, mlp_ratio=float(config["mlp_ratio"]),
            upsampler=str(config["upsampler"]), resi_connection="1conv",
            use_checkpoint=False,
        )
    raise ValueError(f"Unsupported official baseline: {variant}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / "experiments" / cfg["experiment_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "status.json", {"status": "running", "step": 0})
    started = time.perf_counter()

    network = build_network(cfg["model"])
    checkpoint_path = ROOT / cfg["checkpoint"]["path"]
    checkpoint_sha256 = sha256_file(checkpoint_path)
    expected_hash = cfg["model"].get("checkpoint_sha256")
    expected_prefix = cfg["model"].get("checkpoint_sha256_prefix")
    if expected_hash and checkpoint_sha256 != expected_hash:
        raise RuntimeError("Checkpoint SHA256 does not match the audited config")
    if expected_prefix and not checkpoint_sha256.startswith(expected_prefix):
        raise RuntimeError("Checkpoint SHA256 prefix does not match its official filename")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    parameter_key = cfg["checkpoint"].get("parameter_key")
    state_dict = checkpoint[parameter_key] if parameter_key else checkpoint
    # OmniSR's released state includes THOP bookkeeping buffers that are not
    # model parameters and should not participate in loading.
    state_dict = {
        key: value for key, value in state_dict.items()
        if not key.endswith(("total_ops", "total_params"))
    }
    network.load_state_dict(state_dict, strict=True)
    model = GrayscaleViaRGB(
        network, cfg["model"].get("input_range", 1.0)
    ).cuda().eval()
    dataset = LitsSlices(
        cfg["data"]["manifest"], "val", cfg["data"]["val_patch_size"],
        False, cfg["seed"], cfg["data"]["val_slices_per_patient"],
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True,
    )
    totals: defaultdict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            hr = batch["hr"].cuda(non_blocking=True)
            lr = bicubic_downsample(hr, 2)
            sr = model(lr)
            for name, values in batch_metrics(sr, hr, lr, 2).items():
                totals[name].extend(values.cpu().tolist())
            if index == 1 or index % 8 == 0 or index == len(dataset):
                atomic_json(output_dir / "status.json", {
                    "status": "running", "step": index, "total": len(dataset),
                })

    metrics = {name: float(np.mean(values)) for name, values in totals.items()}
    record = {"step": len(dataset), **metrics}
    (output_dir / "validation.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    atomic_json(output_dir / "status.json", {
        "status": "completed", "step": len(dataset),
        "best_psnr": metrics["psnr"], "latest": metrics,
        "checkpoint_mode": "official_rgb_zero_shot_gray_repeat_mean",
        "checkpoint_sha256": checkpoint_sha256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
    })
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
