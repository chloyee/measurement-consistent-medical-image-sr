import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import torch

from data_utils import bicubic_downsample
from util import build_model


def test_selected_parameter_count_and_shapes():
    model = build_model(
        {
            "variant": "safenull_sparse_exchange",
            "scale": 2,
            "channels": 96,
            "depth": 14,
            "exchange_positions": [2, 5],
            "exchange_gate_mode": "learned",
            "dual_window_stem": True,
            "use_exact_projection": True,
            "projection_refinements": 2,
        }
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_759_244
    lr = torch.rand(1, 1, 12, 12)
    output = model(lr, return_diagnostics=True)
    assert output["sr"].shape == (1, 1, 24, 24)
    assert output["exchange_gates"].shape[:3] == (1, 2, 96)
    assert output["scale_evidence"].shape[:3] == (1, 2, 3)


def test_exact_projection_matches_measurement():
    model = build_model(
        {
            "variant": "safenull_sparse_exchange",
            "channels": 8,
            "depth": 4,
            "exchange_positions": [2],
            "dual_window_stem": True,
            "use_exact_projection": True,
            "projection_refinements": 2,
        }
    )
    lr = torch.rand(1, 1, 10, 11)
    sr = model(lr)["sr"]
    error = (bicubic_downsample(sr, 2) - lr).abs().max().item()
    assert error < 2e-5
