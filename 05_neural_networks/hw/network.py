"""HW04 - Neural Networks.

Implement the public API described in spec.pdf:

  - Dropout    custom nn.Module - inverted dropout                  (spec §2)
  - MLP        configurable feedforward network                     (spec §3)

Read the spec end-to-end before you start. It tells you every method signature,
the math for every method, the shape of every input and output, and the
invariants the tests will check.

Restrictions:
  - No torch.nn.Dropout anywhere in this file. Implement Dropout from scratch.
  - No sklearn.neural_network.

Run tests:   uv run pytest
Check score: uv run python score.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Dropout(nn.Module):
    """See spec.pdf §2."""


class MLP(nn.Module):
    """See spec.pdf §3."""
