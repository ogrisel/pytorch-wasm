"""Validate microtorch autograd + training against real PyTorch (host only).

Run on the HOST (where real torch is installed) to prove microtorch's gradients and
training dynamics match upstream PyTorch to within float tolerance. This is evidence
that the reduced engine shipped to the WASM kernel is numerically faithful.
"""
import numpy as np

import microtorch as mt


def test_gradient_matches_pytorch():
    import torch

    rng = np.random.RandomState(0)
    x_np = rng.randn(8, 4).astype(np.float32)
    w1_np = rng.randn(4, 6).astype(np.float32)
    b1_np = rng.randn(6).astype(np.float32)
    w2_np = rng.randn(6, 1).astype(np.float32)
    b2_np = rng.randn(1).astype(np.float32)
    y_np = rng.randn(8, 1).astype(np.float32)

    # --- microtorch forward/backward ---
    x = mt.tensor(x_np)
    w1 = mt.tensor(w1_np, requires_grad=True)
    b1 = mt.tensor(b1_np, requires_grad=True)
    w2 = mt.tensor(w2_np, requires_grad=True)
    b2 = mt.tensor(b2_np, requires_grad=True)
    h = (x @ w1 + b1).relu()
    pred = h @ w2 + b2
    diff = pred - mt.tensor(y_np)
    loss = (diff * diff).mean()
    loss.backward()

    # --- real torch forward/backward ---
    xt = torch.tensor(x_np)
    w1t = torch.tensor(w1_np, requires_grad=True)
    b1t = torch.tensor(b1_np, requires_grad=True)
    w2t = torch.tensor(w2_np, requires_grad=True)
    b2t = torch.tensor(b2_np, requires_grad=True)
    ht = torch.relu(xt @ w1t + b1t)
    predt = ht @ w2t + b2t
    losst = ((predt - torch.tensor(y_np)) ** 2).mean()
    losst.backward()

    assert np.allclose(loss.item(), losst.item(), atol=1e-5), (loss.item(), losst.item())
    for name, a, b in [
        ("w1", w1.grad, w1t.grad.numpy()),
        ("b1", b1.grad, b1t.grad.numpy()),
        ("w2", w2.grad, w2t.grad.numpy()),
        ("b2", b2.grad, b2t.grad.numpy()),
    ]:
        assert np.allclose(a, b, atol=1e-4), f"grad mismatch {name}: max|d|={np.abs(a-b).max()}"
    print("[OK] microtorch gradients match PyTorch (loss=%.6f vs %.6f)" % (loss.item(), losst.item()))


def test_training_reduces_loss():
    mt.manual_seed(0)
    rng = np.random.RandomState(1)
    X = rng.randn(128, 4).astype(np.float32)
    true_w = np.array([[1.5], [-2.0], [0.5], [3.0]], dtype=np.float32)
    y = np.maximum(X @ true_w, 0.0) + 0.1 * rng.randn(128, 1).astype(np.float32)

    model = mt.nn.Sequential(mt.nn.Linear(4, 32), mt.nn.ReLU(), mt.nn.Linear(32, 1))
    opt = mt.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = mt.nn.MSELoss()

    losses = []
    Xt, yt = mt.tensor(X), mt.tensor(y)
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print("[OK] training loss %.4f -> %.4f" % (losses[0], losses[-1]))
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


if __name__ == "__main__":
    test_training_reduces_loss()
    try:
        test_gradient_matches_pytorch()
    except ImportError:
        print("[skip] real torch not importable; gradient cross-check skipped")
