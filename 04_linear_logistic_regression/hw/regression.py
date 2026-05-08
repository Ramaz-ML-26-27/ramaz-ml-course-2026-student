"""HW03 — Linear and Logistic Regression.

Implement the public API described in spec.pdf:

  - sigmoid(z)            — free function (§2)
  - LinearRegression      — class with __init__, forward, loss, gradient, fit, predict (§3)
  - LogisticRegression    — class with __init__, forward, loss, gradient, fit,
                            predict_proba, predict (§4)

Read the spec end-to-end before you start. It tells you every method signature,
the math for every method, the shape of every input and output, and the
invariants the tests will check.

Restrictions:
  - No torch.autograd, no .backward(), no requires_grad=True anywhere in this file.
  - No sklearn.linear_model.

Run tests:   uv run pytest
Check score: uv run python score.py
"""

from __future__ import annotations

import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """See spec.pdf §2."""
    raise NotImplementedError("Implement sigmoid()")


class LinearRegression:
    """See spec.pdf §3."""


class LogisticRegression:
    """See spec.pdf §4."""
