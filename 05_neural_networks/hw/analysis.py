"""HW04 - Analysis: FashionMNIST ablation study.

Implement the two helper functions below (see spec.pdf §7), then run
this script to generate the results you reference in writeup.md:

    uv run python analysis.py

The script trains four configurations of the same MLP on FashionMNIST and
saves the comparison artifacts to the hw/ directory:

    ablation_curves.png   - train/val loss curves for all four configs
    ablation_table.txt    - final accuracies and best-epoch summary

The four configurations are:
    'baseline'        no regularization
    'dropout'         + dropout (p=0.3)
    'batchnorm'       + batch normalization
    'both_schedule'   + dropout + batch norm + weight decay + StepLR schedule
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network import MLP  # noqa: F401  (used by the reference and by tests)

INPUT_DIM: int = 28 * 28
HIDDEN_DIMS: list[int] = [256, 128]
OUTPUT_DIM: int = 10

CONFIGS: tuple[str, ...] = ("baseline", "dropout", "batchnorm", "both_schedule")


def load_fashion_mnist(
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    raise NotImplementedError("Implement load_fashion_mnist()")


def build_model(
    config: str,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    raise NotImplementedError("Implement build_model()")


# ════════════════════════════════════════════════════════════════════════════════
# Below this line is teacher-provided: orchestration, plotting, table generation.
# Do not modify. Run with `uv run python analysis.py`.
# ════════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    from typing import Sized, cast

    import matplotlib.pyplot as plt

    from training import train, validate

    torch.manual_seed(0)

    MAX_EPOCHS = 25
    PATIENCE = 5
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading FashionMNIST...")
    train_loader, val_loader, test_loader = load_fashion_mnist(batch_size=BATCH_SIZE)
    print(
        f"  train: {len(cast(Sized, train_loader.dataset))} examples, "
        f"val: {len(cast(Sized, val_loader.dataset))} examples, "
        f"test: {len(cast(Sized, test_loader.dataset))} examples\n"
    )

    criterion = nn.CrossEntropyLoss()
    histories: dict[str, dict] = {}
    test_accs: dict[str, float] = {}

    for cfg in CONFIGS:
        print(f"=== Training config: {cfg} ===")
        torch.manual_seed(0)  # same init every config
        model, optimizer, scheduler = build_model(cfg)
        model = model.to(device)

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            device=device,
            scheduler=scheduler,
        )
        _, test_acc = validate(model, test_loader, criterion, device)
        histories[cfg] = history
        test_accs[cfg] = test_acc
        print(
            f"  best_epoch={history['best_epoch']}, "
            f"final_train_loss={history['train_loss'][-1]:.4f}, "
            f"final_val_acc={history['val_acc'][-1]:.4f}, "
            f"test_acc={test_acc:.4f}\n"
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {
        "baseline": "tab:blue",
        "dropout": "tab:orange",
        "batchnorm": "tab:green",
        "both_schedule": "tab:red",
    }
    for cfg in CONFIGS:
        h = histories[cfg]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], color=colors[cfg], label=cfg)
        axes[1].plot(epochs, h["val_loss"], color=colors[cfg], label=cfg)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("ablation_curves.png", dpi=150)
    print("Saved: ablation_curves.png")

    # ── Comparison table ─────────────────────────────────────────────────────
    rows: list[str] = []
    rows.append(
        f"{'config':<18} {'best_ep':>7} {'final_train':>12} "
        f"{'final_val_acc':>14} {'test_acc':>10}"
    )
    rows.append("-" * 64)
    for cfg in CONFIGS:
        h = histories[cfg]
        rows.append(
            f"{cfg:<18} "
            f"{h['best_epoch']:>7d} "
            f"{h['train_loss'][-1]:>12.4f} "
            f"{h['val_acc'][-1]:>14.4f} "
            f"{test_accs[cfg]:>10.4f}"
        )
    table = "\n".join(rows)

    with open("ablation_table.txt", "w") as f:
        f.write(table + "\n")
    print("Saved: ablation_table.txt\n")
    print(table)
    print("\nAll done. Use these results to answer the questions in writeup.md.")
