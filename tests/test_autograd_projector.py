"""Numerical checks for the autograd projector. Run with PyTorch 2.4.0."""
import importlib.util
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sparx_under_test", ROOT / "code" / "util.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Projector = module.BicubicExactProjector


def dense_operator(high=(8, 10), low=(4, 5), device="cpu"):
    n = high[0] * high[1]
    basis = torch.eye(n, dtype=torch.double, device=device).reshape(n, 1, *high)
    return Projector.downsample(basis, low).flatten(1).T.contiguous()


def dense_lift(v, high):
    d = dense_operator(high, tuple(v.shape[-2:]), str(v.device))
    b = d.T @ torch.linalg.solve(d @ d.T, torch.eye(d.shape[0], dtype=d.dtype, device=d.device))
    return (v.flatten(-2) @ b.T).reshape(*v.shape[:2], *high)


@pytest.mark.parametrize("high,low", [((8,10),(4,5)), ((9,11),(4,5)), ((10,14),(5,7))])
def test_adjoint_and_dense_agreement(high, low):
    torch.manual_seed(7)
    x = torch.randn(2, 3, *high, dtype=torch.double)
    v = torch.randn(2, 3, *low, dtype=torch.double)
    d = dense_operator(high, low)
    dtv = Projector.adjoint(v, high)
    torch.testing.assert_close((Projector.downsample(x, low)*v).sum(), (x*dtv).sum(), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(dtv.flatten(-2), v.flatten(-2) @ d, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(Projector().lift(v, high), dense_lift(v, high), atol=2e-9, rtol=2e-9)


def test_projection_nullspace_idempotence_and_checkpoint():
    torch.manual_seed(8)
    p = Projector()
    x = torch.randn(2, 1, 8, 10, dtype=torch.double)
    y = torch.randn(2, 1, 4, 5, dtype=torch.double)
    sr = p(x, y)
    torch.testing.assert_close(p.downsample(sr, y.shape[-2:]), y, atol=1e-10, rtol=1e-10)
    null = lambda z: z - dense_lift(p.downsample(z, y.shape[-2:]), (8,10))
    torch.testing.assert_close(null(sr), null(x), atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(p(sr, y), sr, atol=2e-9, rtol=2e-9)
    assert p.state_dict() == {}
    p.load_state_dict({}, strict=True)


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_inference_context(mode):
    with mode():
        x = torch.randn(2, 1, 16, 20)
        y = torch.randn(2, 1, 8, 10)
        sr = Projector()(x, y)
        torch.testing.assert_close(Projector.downsample(sr, y.shape[-2:]), y, atol=2e-6, rtol=2e-6)
        assert not sr.requires_grad


def test_zero_and_mixed_batch():
    y = torch.randn(3,2,5,7, dtype=torch.double)
    y[0] = 0
    y[1] *= 1e-8
    z = Projector().lift(y, (10,14))
    torch.testing.assert_close(Projector.downsample(z, (5,7)), y, atol=1e-10, rtol=1e-9)
    assert torch.count_nonzero(z[0]) == 0


def test_first_order_gradients():
    torch.manual_seed(9)
    p = Projector()
    x = torch.randn(1,1,6,8, dtype=torch.double, requires_grad=True)
    y = torch.randn(1,1,3,4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(p, (x, y), eps=1e-6, atol=2e-5, rtol=2e-4)
    g = torch.randn_like(x)
    dx, dy = torch.autograd.grad((p(x,y)*g).sum(), (x,y))
    d = dense_operator((6,8),(3,4))
    b = d.T @ torch.linalg.solve(d@d.T, torch.eye(12,dtype=torch.double))
    expected_dy = g.flatten(-2) @ b
    expected_dx = g.flatten(-2) @ (torch.eye(48,dtype=torch.double)-b@d)
    torch.testing.assert_close(dx.flatten(-2), expected_dx, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(dy.flatten(-2), expected_dy, atol=2e-8, rtol=2e-8)


def test_training_step():
    model = torch.nn.Conv2d(1,1,3,padding=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2,1,16,20)
    y = torch.randn(2,1,8,10)
    old = model.weight.detach().clone()
    sr = Projector()(model(x), y)
    sr.square().mean().backward()
    assert torch.isfinite(model.weight.grad).all()
    optimizer.step()
    assert not torch.equal(old, model.weight)


def test_nonconvergence_and_nonfinite():
    with pytest.raises(RuntimeError, match="converge"):
        Projector(max_iter=1, rtol=1e-12, atol=1e-14).lift(torch.randn(1,1,8,10,dtype=torch.double),(16,20))
    with pytest.raises(ValueError, match="non-finite"):
        Projector().lift(torch.full((1,1,4,5),float("nan")),(8,10))


def test_512_output_cpu():
    torch.manual_seed(10)
    with torch.inference_mode():
        x = torch.randn(1,1,512,512)
        y = torch.rand(1,1,256,256)
        sr = Projector()(x,y)
        err = (Projector.downsample(sr,(256,256))-y).abs().max().item()
        assert err < 2e-6, err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA server required")
def test_cuda_forward_backward_and_inference():
    p = Projector().cuda()
    x = torch.randn(1,1,64,64,device="cuda",requires_grad=True)
    y = torch.rand(1,1,32,32,device="cuda")
    sr = p(x,y)
    sr.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    with torch.inference_mode():
        sr = p(x,y)
        torch.testing.assert_close(p.downsample(sr,(32,32)),y,atol=2e-6,rtol=2e-6)
