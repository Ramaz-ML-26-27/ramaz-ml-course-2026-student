"""Pytest configuration for HW04.

Sets deterministic seeds for reproducibility, and enforces that network.py
does not use torch.nn.Dropout (students must implement Dropout from scratch
as the lecture and spec describe).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

# ── Deterministic seeds ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Set deterministic seeds before every test."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True


# ── Source loading ─────────────────────────────────────────────────────────────


_NETWORK_SOURCE = (Path(__file__).parent / "network.py").read_text()


# ── Block torch.nn.Dropout in network.py (AST-based, not substring) ────────────
# Students implement Dropout from scratch; using nn.Dropout bypasses the
# learning goal in spec §2. AST walking means we don't trip on the literal
# string "nn.Dropout" appearing in a docstring or comment.


def _uses_torch_nn_dropout(source: str) -> bool:
    """Detect any reference to torch.nn.Dropout in network.py source.

    Catches:
      - nn.Dropout(...)
      - torch.nn.Dropout(...)
      - from torch.nn import Dropout
      - from torch.nn import Dropout as Foo
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # from torch.nn import Dropout (any alias)
        if isinstance(node, ast.ImportFrom) and node.module == "torch.nn":
            for alias in node.names:
                if alias.name == "Dropout":
                    return True

        # nn.Dropout - attribute access on a Name 'nn'
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "nn"
            and node.attr == "Dropout"
        ):
            return True

        # torch.nn.Dropout - attribute access on (torch.nn)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "torch"
            and node.value.attr == "nn"
            and node.attr == "Dropout"
        ):
            return True

    return False


@pytest.fixture(autouse=True)
def block_nn_dropout() -> None:
    """Fail if network.py uses torch.nn.Dropout anywhere."""
    if _uses_torch_nn_dropout(_NETWORK_SOURCE):
        pytest.fail(
            "network.py must NOT use torch.nn.Dropout. "
            "Implement Dropout from scratch as a subclass of nn.Module, "
            "using self.training and the inverted-dropout scaling derived "
            "in lecture (spec.pdf §2)."
        )


# ── Block sklearn.neural_network (would bypass the assignment) ─────────────────


def _uses_sklearn_neural_network(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sklearn.neural_network" or alias.name.startswith(
                    "sklearn.neural_network."
                ):
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "sklearn.neural_network"
                or node.module.startswith("sklearn.neural_network.")
            ):
                return True
    return False


@pytest.fixture(autouse=True)
def block_sklearn_neural_network() -> None:
    """Fail if network.py imports sklearn.neural_network."""
    if _uses_sklearn_neural_network(_NETWORK_SOURCE):
        pytest.fail(
            "network.py must NOT import sklearn.neural_network. "
            "Implement MLP from scratch using torch.nn building blocks."
        )
