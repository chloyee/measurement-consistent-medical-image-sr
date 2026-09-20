"""Core SPARX modules used by the paper experiments.

This module intentionally contains only the selected C96D14 family and the
ablation switches needed by the released tables. Historical candidate trunks
are not part of the public API.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class CrossScaleLargeFieldBlock(nn.Module):
    """Content-routed compact, axial and sparse large-field experts."""

    def __init__(self, channels: int):
        super().__init__()
        self.compact = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels
        )
        self.axial = nn.Sequential(
            nn.Conv2d(
                channels, channels, (1, 15), padding=(0, 7), groups=channels
            ),
            nn.Conv2d(
                channels, channels, (15, 1), padding=(7, 0), groups=channels
            ),
        )
        self.sparse = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=2, dilation=2, groups=channels
            ),
            nn.Conv2d(
                channels, channels, 3, padding=4, dilation=4, groups=channels
            ),
        )
        self.evidence = nn.Conv2d(channels, 3, 1)
        self.mix = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU())
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branches = torch.stack(
            (self.compact(x), self.axial(x), self.sparse(x)), dim=1
        )
        evidence = torch.softmax(self.evidence(x), dim=1)
        routed = (branches * evidence.unsqueeze(2)).sum(dim=1)
        output = x + torch.tanh(self.residual_scale) * self.mix(routed)
        return output, evidence


class SparseLargeFieldExchange(nn.Module):
    """Inject a bounded expert residual at a sparse backbone location."""

    def __init__(
        self,
        channels: int,
        exchange_scale_init: float = 1.0,
        gate_mode: str = "learned",
    ):
        super().__init__()
        if gate_mode not in {"learned", "ungated"}:
            raise ValueError(f"unknown gate mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.large_field = CrossScaleLargeFieldBlock(channels)
        self.field_gate = (
            nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
            if gate_mode == "learned"
            else None
        )
        self.mix = nn.Conv2d(channels, channels, 1)
        nn.init.dirac_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.exchange_scale = nn.Parameter(torch.tensor(exchange_scale_init))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        field_feature, scale_evidence = self.large_field(x)
        if self.field_gate is None:
            field_gate = torch.ones_like(x)
        else:
            field_gate = self.field_gate(torch.cat((x, field_feature), dim=1))
        update = self.mix(field_gate * (field_feature - x))
        output = x + torch.tanh(self.exchange_scale) * update
        return output, field_gate, scale_evidence, field_feature


class SparseExchangeBackbone(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        exchange_positions: tuple[int, ...],
        exchange_scale_init: float,
        exchange_gate_mode: str,
    ):
        super().__init__()
        if any(position < 1 or position >= depth for position in exchange_positions):
            raise ValueError("exchange positions must lie inside the backbone")
        if tuple(sorted(set(exchange_positions))) != exchange_positions:
            raise ValueError("exchange positions must be sorted and unique")
        self.exchange_positions = exchange_positions
        self.local_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(depth)]
        )
        self.exchanges = nn.ModuleList(
            [
                SparseLargeFieldExchange(
                    channels, exchange_scale_init, exchange_gate_mode
                )
                for _ in exchange_positions
            ]
        )
        self.long_mix = nn.Conv2d(channels, channels, 1)
        nn.init.dirac_(self.long_mix.weight)
        nn.init.zeros_(self.long_mix.bias)
        self.long_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, shallow: torch.Tensor) -> dict[str, torch.Tensor]:
        x = shallow
        gates: list[torch.Tensor] = []
        evidence_maps: list[torch.Tensor] = []
        field_features: list[torch.Tensor] = []
        exchange_index = 0
        for block_index, block in enumerate(self.local_blocks, start=1):
            x = block(x)
            if block_index in self.exchange_positions:
                x, gate, evidence, field = self.exchanges[exchange_index](x)
                gates.append(gate)
                evidence_maps.append(evidence)
                field_features.append(field)
                exchange_index += 1
        feature = shallow + torch.tanh(self.long_scale) * self.long_mix(x - shallow)
        empty = x.new_empty((x.shape[0], 0, x.shape[1], *x.shape[-2:]))
        return {
            "feature": feature,
            "local": x,
            "global": field_features[-1] if field_features else x,
            "gate": gates[-1] if gates else x.new_empty(0),
            "exchange_gates": torch.stack(gates, dim=1) if gates else empty,
            "scale_evidence": (
                torch.stack(evidence_maps, dim=1) if evidence_maps else empty
            ),
        }


@lru_cache(maxsize=16)
def _bicubic_right_inverse_1d(high_size: int, low_size: int) -> torch.Tensor:
    basis = torch.eye(high_size, dtype=torch.float64).view(high_size, 1, 1, high_size)
    operator = F.interpolate(
        basis,
        size=(1, low_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[:, 0, 0, :].transpose(0, 1).contiguous()
    gram = operator @ operator.transpose(0, 1)
    identity = torch.eye(low_size, dtype=torch.float64)
    return (operator.transpose(0, 1) @ torch.linalg.solve(gram, identity)).contiguous()


class BicubicExactProjector(nn.Module):
    """Project an HR estimate onto the affine measurement set D(x)=LR."""

    def __init__(self, scale: int = 2, refinement_steps: int = 2):
        super().__init__()
        self.scale = int(scale)
        self.refinement_steps = int(refinement_steps)
        self._device_cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def _right_inverse(
        self, high_size: int, low_size: int, reference: torch.Tensor
    ) -> torch.Tensor:
        key = (high_size, low_size, str(reference.device), reference.dtype)
        matrix = self._device_cache.get(key)
        if matrix is None:
            matrix = _bicubic_right_inverse_1d(high_size, low_size).to(
                device=reference.device, dtype=reference.dtype
            )
            self._device_cache[key] = matrix
        return matrix

    @staticmethod
    def downsample(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            image, size=size, mode="bicubic", align_corners=False, antialias=True
        )

    def lift(self, low: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        inverse_h = self._right_inverse(size[0], low.shape[-2], low)
        inverse_w = self._right_inverse(size[1], low.shape[-1], low)
        return inverse_h @ low @ inverse_w.transpose(0, 1)

    def forward(self, candidate: torch.Tensor, measurement: torch.Tensor) -> torch.Tensor:
        projected = candidate
        for _ in range(self.refinement_steps):
            residual = measurement - self.downsample(
                projected, measurement.shape[-2:]
            )
            projected = projected + self.lift(residual, candidate.shape[-2:])
        return projected


class SPARX(nn.Module):
    """Measurement-consistent SR with gated sparse large-field experts."""

    def __init__(
        self,
        scale: int = 2,
        channels: int = 96,
        depth: int = 14,
        exchange_positions: tuple[int, ...] = (2, 5),
        projection_refinements: int = 2,
        exchange_scale_init: float = 1.0,
        exchange_gate_mode: str = "learned",
        use_exact_projection: bool = True,
        dual_window_stem: bool = True,
    ):
        super().__init__()
        self.scale = int(scale)
        self.dual_window_stem = bool(dual_window_stem)
        self.runtime_window_domain = False
        self.shallow = nn.Conv2d(1, channels, 3, padding=1)
        self.liver_shallow = (
            nn.Conv2d(1, channels, 3, padding=1) if dual_window_stem else None
        )
        self.backbone = SparseExchangeBackbone(
            channels,
            depth,
            exchange_positions,
            exchange_scale_init,
            exchange_gate_mode,
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.projector = BicubicExactProjector(scale, projection_refinements)
        self.use_exact_projection = bool(use_exact_projection)

    def forward(
        self,
        lr: torch.Tensor,
        return_diagnostics: bool = False,
        window_domain: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = F.interpolate(
            lr,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        wide_shallow = self.shallow(lr)
        shallow = wide_shallow
        liver_shallow = torch.empty(0, device=lr.device, dtype=lr.dtype)
        if self.dual_window_stem:
            assert self.liver_shallow is not None
            liver_shallow = self.liver_shallow(lr)
            if window_domain is None:
                value = 1.0 if self.runtime_window_domain else 0.0
                window_domain = lr.new_full((lr.shape[0], 1, 1, 1), value)
            shallow = torch.lerp(wide_shallow, liver_shallow, window_domain)
        features = self.backbone(shallow)
        prior_residual = self.reconstruction(features["feature"])
        candidate = base + prior_residual
        sr = self.projector(candidate, lr) if self.use_exact_projection else candidate
        output = {
            "sr": sr,
            "preliminary_sr": candidate,
            "gate": features["gate"],
            "flow": torch.empty(0, device=lr.device),
        }
        if return_diagnostics:
            range_component = self.projector.lift(lr, sr.shape[-2:])
            output.update(
                {
                    "shallow": shallow,
                    "wide_shallow": wide_shallow,
                    "liver_shallow": liver_shallow,
                    "local": features["local"],
                    "global": features["global"],
                    "reconstruction_feature": features["feature"],
                    "fused_before_sfft": features["feature"],
                    "fused": features["feature"],
                    "range_component": range_component,
                    "null_detail": sr - range_component,
                    "prior_residual": prior_residual,
                    "scale_evidence": features["scale_evidence"],
                    "exchange_gates": features["exchange_gates"],
                }
            )
        return output

    def set_input_window(self, center: float, width: float, quantization_hu: float) -> None:
        del center, width, quantization_hu
        self.runtime_window_domain = True

    def initialize_liver_stem_from_wide(self) -> None:
        if self.liver_shallow is not None:
            self.liver_shallow.load_state_dict(self.shallow.state_dict())


def build_model(config: dict) -> SPARX:
    if config.get("variant") not in {"sparx", "safenull_sparse_exchange"}:
        raise ValueError("the review package exposes only the selected SPARX family")
    return SPARX(
        scale=config.get("scale", 2),
        channels=config.get("channels", 96),
        depth=config.get("depth", 14),
        exchange_positions=tuple(config.get("exchange_positions", (2, 5))),
        projection_refinements=config.get("projection_refinements", 2),
        exchange_scale_init=config.get("exchange_scale_init", 1.0),
        exchange_gate_mode=config.get("exchange_gate_mode", "learned"),
        use_exact_projection=config.get("use_exact_projection", True),
        dual_window_stem=config.get("dual_window_stem", True),
    )
