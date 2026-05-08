"""HW03 Pipeline Demo — sanity check for your regression classes.

Run this as soon as you have implemented LinearRegression and LogisticRegression
in regression.py:

    uv run python pipeline.py

You should see:
  - LinearRegression training loss decrease, recovered weights close to the truth
  - LogisticRegression reaching >95% accuracy on a separable problem

If those numbers look good, your full pipeline works end-to-end. The unit tests
in test_regression.py go deeper — but this is a fast smoke test.

This file is read-only — you do not need to edit it.
"""

from __future__ import annotations

import torch

from regression import LinearRegression, LogisticRegression


def demo_linear() -> None:
    """Train LinearRegression on synthetic data y = 2·x_1 - x_2 + 0.5."""
    print("=" * 60)
    print("LinearRegression  —  synthetic, y = 2·x_1 - x_2 + 0.5 + noise")
    print("=" * 60)

    torch.manual_seed(0)
    n = 200
    X = torch.randn(n, 2)
    w_true = torch.tensor([2.0, -1.0])
    b_true = 0.5
    y = X @ w_true + b_true + 0.05 * torch.randn(n)

    model = LinearRegression(n_features=2, lr=0.05)
    history = model.fit(X, y, epochs=200)

    print(f"Initial loss: {history[0]:.4f}")
    print(f"Final loss:   {history[-1]:.4f}")
    print(
        f"Learned w:    [{model.w[0].item():.4f}, {model.w[1].item():.4f}]"
        f"   (true: [{w_true[0].item():.1f}, {w_true[1].item():.1f}])"
    )
    print(f"Learned b:    {model.b.item():.4f}    (true: {b_true})")
    print()


def demo_logistic() -> None:
    """Train LogisticRegression on a synthetic, linearly-separable problem."""
    print("=" * 60)
    print("LogisticRegression — synthetic, linearly separable")
    print("=" * 60)

    torch.manual_seed(0)
    n = 300
    X = torch.randn(n, 2)
    w_true = torch.tensor([1.5, -0.8])
    b_true = 0.3
    logits = X @ w_true + b_true
    y = (logits > 0).float()

    model = LogisticRegression(n_features=2, lr=0.5)
    history = model.fit(X, y, epochs=400)

    accuracy = (model.predict(X) == y).float().mean().item()
    print(f"Initial loss: {history[0]:.4f}")
    print(f"Final loss:   {history[-1]:.4f}")
    print(f"Accuracy:     {accuracy * 100:.1f}%")
    print(
        f"Learned w:    [{model.w[0].item():.4f}, {model.w[1].item():.4f}]"
        f"   (true: [{w_true[0].item():.1f}, {w_true[1].item():.1f}])"
    )
    print(f"Learned b:    {model.b.item():.4f}    (true: {b_true})")
    print()


if __name__ == "__main__":
    demo_linear()
    demo_logistic()
    print("Pipeline OK.  Now run 'uv run pytest' for the full grade-bearing tests.")
