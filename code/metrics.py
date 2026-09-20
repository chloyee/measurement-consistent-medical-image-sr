from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import bicubic_downsample


def batch_metrics(sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor, scale: int) -> dict[str, torch.Tensor]:
    mse = (sr - hr).square().flatten(1).mean(1).clamp_min(1e-12)
    psnr = -10 * torch.log10(mse)
    mae_hu = (sr - hr).abs().flatten(1).mean(1) * 3024
    dc_hu = (bicubic_downsample(sr, scale) - lr).abs().flatten(1).mean(1) * 3024
    mu_x, mu_y = F.avg_pool2d(sr, 11, 1, 5), F.avg_pool2d(hr, 11, 1, 5)
    var_x = F.avg_pool2d(sr * sr, 11, 1, 5) - mu_x.square()
    var_y = F.avg_pool2d(hr * hr, 11, 1, 5) - mu_y.square()
    cov = F.avg_pool2d(sr * hr, 11, 1, 5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2) /
            ((mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2))).flatten(1).mean(1)
    return {"psnr": psnr, "ssim": ssim, "mae_hu": mae_hu, "data_consistency_hu": dc_hu}
