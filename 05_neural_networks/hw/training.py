"""HW04 - Training utilities.

Implement the public API described in spec.pdf:

  - train_one_epoch(model, loader, criterion, optimizer, device)    (spec §4)
  - validate(model, loader, criterion, device)                      (spec §5)
  - train(model, train_loader, val_loader, optimizer, criterion,
          max_epochs, patience, device, scheduler=None)             (spec §6)

Read the spec end-to-end before you start.

Run tests: uv run pytest
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    raise NotImplementedError("Implement train_one_epoch()")


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
) -> tuple[float, float]:
    raise NotImplementedError("Implement validate()")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_epochs: int,
    patience: int,
    device: torch.device | str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    raise NotImplementedError("Implement train()")
