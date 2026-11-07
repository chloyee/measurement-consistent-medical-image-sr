"""User-provided slice data and matched bicubic degradation."""
from __future__ import annotations
import csv
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def bicubic_downsample(hr, scale=2):
    if any(n % scale for n in hr.shape[-2:]):
        raise ValueError("HR dimensions must be divisible by scale")
    return F.interpolate(hr, size=tuple(n // scale for n in hr.shape[-2:]),
                         mode="bicubic", align_corners=False, antialias=True)


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"patient_id", "partition", "image_path"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires patient_id, partition, image_path")
        rows = list(reader)
    patients, images = {}, set()
    for row in rows:
        patient, partition = row["patient_id"].strip(), row["partition"].strip()
        if not patient or partition not in {"train", "val", "test"}:
            raise ValueError("Use nonempty patient IDs and train/val/test partitions")
        if patient in patients and patients[patient] != partition:
            raise ValueError("A patient occurs in multiple partitions")
        patients[patient] = partition
        image = Path(row["image_path"])
        image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
        if image in images:
            raise ValueError("Duplicate image path in manifest")
        images.add(image)
        row.update(patient_id=patient, partition=partition, image_path=str(image))
        row["series_id"] = row.get("series_id") or patient
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def load_slice(path):
    path = Path(path)
    lower, width = 0.0, 0.0
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as item:
            image = item["image"].astype(np.float32)
            if "lower_hu" in item and "width_hu" in item:
                lower, width = float(item["lower_hu"]), float(item["width_hu"])
    elif path.suffix == ".npy":
        image = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raise ValueError("Use .npy or .npz slices")
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Each slice must be a finite two-dimensional array")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Store normalized reference intensities in [0, 1]")
    if not np.isfinite(lower) or not np.isfinite(width) or width < 0:
        raise ValueError("Invalid intensity metadata")
    return torch.from_numpy(image.copy()).unsqueeze(0), lower, width


class SliceDataset(Dataset):
    def __init__(self, manifest, partition, patch_size=None, training=False):
        if training and partition != "train":
            raise ValueError("Optimization uses the train partition")
        self.rows = [r for r in read_manifest(manifest) if r["partition"] == partition]
        if not self.rows:
            raise ValueError(f"Empty partition: {partition}")
        self.patch_size, self.training = patch_size, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image, lower, width = load_slice(row["image_path"])
        h, w = image.shape[-2:]
        if self.patch_size:
            p = int(self.patch_size)
            if p % 2 or p < 2 or min(h, w) < p:
                raise ValueError("Use an even patch size that fits each image")
            top = random.randint(0, h-p) if self.training else (h-p)//2
            left = random.randint(0, w-p) if self.training else (w-p)//2
            image = image[:, top:top+p, left:left+p]
        elif h % 2 or w % 2:
            raise ValueError("Full images must have even height and width")
        if self.training:
            if random.random() < .5:
                image = image.flip(-1)
            if random.random() < .5:
                image = image.flip(-2)
            image = torch.rot90(image, random.randrange(4), (-2, -1))
        return {"hr": image.contiguous(), "patient_id": row["patient_id"],
                "series_id": row["series_id"], "lower_hu": lower, "width_hu": width}


def mix_liver_window_views(image, probability, lower_hu, width_hu):
    if not 0 <= probability <= 1:
        raise ValueError("View probability must be in [0,1]")
    domain = image.new_zeros((image.shape[0], 1, 1, 1))
    if probability == 0:
        return image, domain
    lower = torch.as_tensor(lower_hu, device=image.device, dtype=image.dtype).reshape(-1,1,1,1)
    width = torch.as_tensor(width_hu, device=image.device, dtype=image.dtype).reshape(-1,1,1,1)
    if torch.any(width <= 0):
        raise ValueError("Liver-window augmentation requires lower_hu and positive width_hu per slice")
    domain = (torch.rand_like(domain) < probability).to(image.dtype)
    window = ((image * width + lower + 20.0) / 120.0).clamp(0,1)
    return torch.where(domain.bool(), window, image), domain
