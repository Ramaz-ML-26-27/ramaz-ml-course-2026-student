"""HW03 — Analysis: Regression on real datasets.

Implement the four helper functions described in spec.pdf §5, then run

    uv run python analysis.py

to generate the plots you'll reference in writeup.md.

Prerequisites: complete regression.py first.
"""

from __future__ import annotations

import torch

from regression import LinearRegression, LogisticRegression  # noqa: F401


def load_regression_data() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    raise NotImplementedError("Implement load_regression_data()")


def train_linear_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 200,
) -> tuple["LinearRegression", list[float]]:
    raise NotImplementedError("Implement train_linear_model()")


def load_classification_data() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    raise NotImplementedError("Implement load_classification_data()")


def train_logistic_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.5,
    epochs: int = 300,
) -> tuple["LogisticRegression", list[float]]:
    raise NotImplementedError("Implement train_logistic_model()")


if __name__ == "__main__":
    print(
        "analysis.py is not yet implemented. "
        "Implement the four helper functions above, then re-run."
    )
