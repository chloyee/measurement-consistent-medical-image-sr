from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def bicubic_downsample(hr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False, antialias=True)


def liver_window_view(
    image: torch.Tensor, source_lower_hu: float = -1024.0,
    source_width_hu: float = 3024.0, window_center_hu: float = 40.0,
    window_width_hu: float = 120.0,
) -> torch.Tensor:
    """Map normalized source-window pixels into the confirmed HKU liver window."""
    if source_width_hu <= 0 or window_width_hu <= 0:
        raise ValueError("window widths must be positive")
    hu = image * source_width_hu + source_lower_hu
    window_lower = window_center_hu - window_width_hu / 2.0
    return ((hu - window_lower) / window_width_hu).clamp(0.0, 1.0)


def mix_liver_window_views(
    image: torch.Tensor, probability: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace independent batch samples with liver-window views before LR synthesis."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("window-mix probability must be inside [0, 1]")
    if probability == 0.0:
        return image, image.new_zeros((image.shape[0], 1, 1, 1))
    select = torch.rand(
        image.shape[0], 1, 1, 1, device=image.device
    ) < probability
    mixed = torch.where(select, liver_window_view(image), image)
    return mixed, select.to(image.dtype)


class LitsSlices(Dataset):
    def __init__(self, manifest: str, partition: str, patch_size: int | None, training: bool, seed: int = 2026,
                 val_slices_per_patient: int = 8):
        with open(manifest, newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["partition"] == partition]
        self.training = training
        self.patch_size = patch_size
        self.seed = seed
        if training:
            liver = [r for r in rows if int(r["liver_pixels"]) > 0]
            non_liver = [r for r in rows if int(r["liver_pixels"]) == 0]
            rng = random.Random(seed)
            rng.shuffle(non_liver)
            rows = liver + non_liver[: max(1, len(liver) // 3)]
        else:
            selected = []
            patients = sorted({r["patient_id"] for r in rows})
            for patient in patients:
                candidates = [r for r in rows if r["patient_id"] == patient and int(r["liver_pixels"]) > 0]
                if not candidates:
                    candidates = [r for r in rows if r["patient_id"] == patient]
                positions = np.linspace(0, len(candidates) - 1, min(val_slices_per_patient, len(candidates)), dtype=int)
                selected.extend(candidates[i] for i in positions)
            rows = selected
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        stored = np.asarray(Image.open(row["image_path"]), dtype=np.int32)
        hr = torch.from_numpy(stored.astype(np.float32) / 3024.0).unsqueeze(0)
        if self.patch_size is not None:
            _, height, width = hr.shape
            size = self.patch_size
            if height < size or width < size:
                raise ValueError(f"Image smaller than patch {size}: {row['image_path']}")
            if self.training:
                top = random.randint(0, height - size)
                left = random.randint(0, width - size)
            else:
                top, left = (height - size) // 2, (width - size) // 2
            hr = hr[:, top:top + size, left:left + size]
            if self.training:
                if random.random() < 0.5: hr = hr.flip(-1)
                if random.random() < 0.5: hr = hr.flip(-2)
                hr = torch.rot90(hr, random.randrange(4), (-2, -1))
        return {"hr": hr.contiguous(), "patient_id": row["patient_id"], "slice_index": int(row["slice_index"])}
