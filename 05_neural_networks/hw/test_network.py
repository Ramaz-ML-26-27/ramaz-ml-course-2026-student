"""Tests for HW04 - network.py, training.py, analysis.py.

Tests are organized in three layers (see spec.pdf section 8):

    @pytest.mark.unit         - one method/class at a time   (35 pts)
    @pytest.mark.integration  - full training pipelines      (30 pts)
    @pytest.mark.analysis     - analysis.py helpers          (35 pts)

Each test class corresponds to one scoring bucket. Every test in a class must
pass to earn the bucket's credit.

Run with: uv run pytest
"""

from __future__ import annotations

from typing import Sized, cast

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from analysis import (
    CONFIGS,
    HIDDEN_DIMS,
    INPUT_DIM,
    OUTPUT_DIM,
    build_model,
    load_fashion_mnist,
)
from network import MLP, Dropout
from training import train, train_one_epoch, validate

TOL = 1e-4


# ── Helpers ────────────────────────────────────────────────────────────────────


def _tiny_classification_loader(
    n: int = 64, d: int = 8, n_classes: int = 4, batch_size: int = 16, seed: int = 0
) -> DataLoader:
    """A small synthetic classification dataset for integration tests."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    # Make labels depend on the first feature so the model can learn something.
    y = (X[:, 0] * 2 + X[:, 1]).long() % n_classes
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def _params_snapshot(model: nn.Module) -> list[torch.Tensor]:
    """Return a deep-copied list of parameter tensors for comparison."""
    return [p.detach().clone() for p in model.parameters()]


def _params_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    """True if any parameter tensor differs between snapshots."""
    return any(not torch.allclose(b, a, atol=TOL) for b, a in zip(before, after))


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests - 35 pts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDropoutTrainingMode:
    """Dropout in training mode. spec.pdf section 2."""

    def test_p_stored(self) -> None:
        d = Dropout(p=0.3)
        assert abs(d.p - 0.3) < TOL, "Dropout must store the rate as self.p"

    def test_shape_preserved(self) -> None:
        d = Dropout(p=0.5)
        d.train()
        x = torch.randn(4, 7)
        assert d(x).shape == x.shape, "Dropout output must match input shape"

    def test_zeros_present_in_training(self) -> None:
        d = Dropout(p=0.5)
        d.train()
        x = torch.ones(10000)
        out = d(x)
        assert (out == 0).any(), "In training mode with p=0.5, some entries should be zero"

    def test_expected_zero_fraction(self) -> None:
        d = Dropout(p=0.3)
        d.train()
        x = torch.ones(20000)
        out = d(x)
        frac_zero = (out == 0).float().mean().item()
        assert abs(frac_zero - 0.3) < 0.05, (
            f"Fraction of zeros should be approximately p=0.3, got {frac_zero:.3f}"
        )

    def test_p_zero_returns_input(self) -> None:
        d = Dropout(p=0.0)
        d.train()
        x = torch.randn(8, 5)
        out = d(x)
        assert torch.allclose(out, x, atol=TOL), (
            "With p=0, Dropout should be the identity even in training mode"
        )

    def test_is_nn_module(self) -> None:
        d = Dropout(p=0.5)
        assert isinstance(d, nn.Module), "Dropout must subclass nn.Module"


@pytest.mark.unit
class TestDropoutEvalMode:
    """Dropout in eval mode. spec.pdf section 2."""

    def test_identity_in_eval(self) -> None:
        d = Dropout(p=0.5)
        d.eval()
        x = torch.randn(20, 10)
        out = d(x)
        assert torch.allclose(out, x, atol=TOL), (
            "In eval mode, Dropout must return its input unchanged"
        )

    def test_no_zeros_added_in_eval(self) -> None:
        d = Dropout(p=0.5)
        d.eval()
        x = torch.ones(1000)
        out = d(x)
        assert (out != 0).all(), "In eval mode, no entries should be zero (no dropping happens)"

    def test_switch_back_to_train(self) -> None:
        d = Dropout(p=0.5)
        d.eval()
        x = torch.ones(1000)
        _ = d(x)
        d.train()
        out = d(x)
        assert (out == 0).any(), "After switching back to train mode, dropout should drop again"


@pytest.mark.unit
class TestDropoutScaling:
    """Inverted-dropout scaling preserves expected activation. spec.pdf section 2."""

    def test_nonzero_entries_scaled(self) -> None:
        d = Dropout(p=0.4)
        d.train()
        x = torch.ones(5000) * 3.0
        out = d(x)
        nonzero = out[out != 0]
        expected = 3.0 / (1.0 - 0.4)
        assert torch.allclose(nonzero, torch.full_like(nonzero, expected), atol=TOL), (
            f"Surviving entries must be scaled by 1/(1-p). "
            f"Expected {expected:.4f}, got {nonzero.unique().tolist()}"
        )

    def test_expectation_preserved(self) -> None:
        d = Dropout(p=0.5)
        d.train()
        x = torch.full((20000,), 4.0)
        out = d(x)
        mean = out.mean().item()
        assert abs(mean - 4.0) < 0.1, (
            f"E[Dropout(x)] should equal x. Expected 4.0, got mean {mean:.3f}"
        )

    def test_different_p_different_scale(self) -> None:
        d = Dropout(p=0.2)
        d.train()
        x = torch.ones(5000) * 2.0
        out = d(x)
        nonzero = out[out != 0]
        expected = 2.0 / (1.0 - 0.2)
        assert torch.allclose(nonzero, torch.full_like(nonzero, expected), atol=TOL), (
            f"With p=0.2, surviving entries should equal x/(1-0.2)={expected:.4f}"
        )


@pytest.mark.unit
class TestMLPArchitecture:
    """MLP layer composition. spec.pdf section 3."""

    def _count(self, model: nn.Module, layer_type: type) -> int:
        return sum(1 for m in model.modules() if isinstance(m, layer_type))

    def test_is_nn_module(self) -> None:
        m = MLP(input_dim=4, hidden_dims=[8], output_dim=2)
        assert isinstance(m, nn.Module), "MLP must subclass nn.Module"

    def test_baseline_layer_counts(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        assert self._count(m, nn.Linear) == 3, (
            "MLP([256,128] -> 10) must have 3 nn.Linear layers (2 hidden + 1 output)"
        )
        assert self._count(m, nn.Flatten) == 1, "MLP must use exactly one nn.Flatten"
        assert self._count(m, nn.BatchNorm1d) == 0, (
            "Baseline MLP (use_batchnorm=False) must have no BatchNorm1d layers"
        )
        assert self._count(m, Dropout) == 0, (
            "Baseline MLP (dropout_p=0) must have no Dropout layers"
        )
        assert self._count(m, nn.ReLU) == 2, "MLP must use one ReLU per hidden layer"

    def test_with_dropout(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10, dropout_p=0.3)
        assert self._count(m, Dropout) == 2, (
            "With dropout_p>0, MLP must have one custom Dropout per hidden layer"
        )
        for d in m.modules():
            if isinstance(d, Dropout):
                assert abs(d.p - 0.3) < TOL, (
                    f"Each Dropout must use the configured p value. Got p={d.p}"
                )

    def test_with_batchnorm(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10, use_batchnorm=True)
        assert self._count(m, nn.BatchNorm1d) == 2, (
            "With use_batchnorm=True, MLP must have one nn.BatchNorm1d per hidden layer"
        )

    def test_linear_dimensions(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        linears = [layer for layer in m.modules() if isinstance(layer, nn.Linear)]
        dims = [(layer.in_features, layer.out_features) for layer in linears]
        # Order may depend on iteration order but the multiset must match.
        expected = [(784, 256), (256, 128), (128, 10)]
        assert sorted(dims) == sorted(expected), (
            f"Linear (in, out) dimensions must be {expected}, got {dims}"
        )

    def test_with_both_regularizers(self) -> None:
        m = MLP(
            input_dim=784,
            hidden_dims=[256, 128],
            output_dim=10,
            dropout_p=0.3,
            use_batchnorm=True,
        )
        assert self._count(m, Dropout) == 2
        assert self._count(m, nn.BatchNorm1d) == 2
        assert self._count(m, nn.Linear) == 3


@pytest.mark.unit
class TestMLPForwardShapes:
    """MLP forward pass produces correct output shapes. spec.pdf section 3."""

    def test_flattens_image_input(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[64], output_dim=10)
        m.eval()
        x = torch.randn(4, 1, 28, 28)
        assert m(x).shape == (4, 10), (
            f"MLP must flatten (B, 1, 28, 28) inputs and output (B, 10). Got {tuple(m(x).shape)}"
        )

    def test_batch_size_one(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[64], output_dim=10)
        m.eval()
        x = torch.randn(1, 1, 28, 28)
        assert m(x).shape == (1, 10), "MLP must handle batch size 1"

    def test_batch_size_32(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[64], output_dim=10)
        m.eval()
        x = torch.randn(32, 1, 28, 28)
        assert m(x).shape == (32, 10), "MLP must handle batch size 32"

    def test_with_batchnorm_runs(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[64], output_dim=10, use_batchnorm=True)
        m.eval()
        x = torch.randn(4, 1, 28, 28)
        assert m(x).shape == (4, 10), "MLP with batchnorm must produce correct shape"

    def test_with_dropout_runs(self) -> None:
        m = MLP(input_dim=784, hidden_dims=[64], output_dim=10, dropout_p=0.3)
        m.eval()
        x = torch.randn(4, 1, 28, 28)
        assert m(x).shape == (4, 10), "MLP with dropout must produce correct shape"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests - 30 pts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestTrainOneEpoch:
    """train_one_epoch end-to-end. spec.pdf section 4."""

    def test_returns_float(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        loader = _tiny_classification_loader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        out = train_one_epoch(model, loader, criterion, optimizer, "cpu")
        assert isinstance(out, float), (
            f"train_one_epoch must return a Python float, got {type(out).__name__}"
        )

    def test_parameters_change(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        loader = _tiny_classification_loader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        before = _params_snapshot(model)
        train_one_epoch(model, loader, criterion, optimizer, "cpu")
        after = _params_snapshot(model)
        assert _params_changed(before, after), (
            "train_one_epoch must update parameters (optimizer.step() must be called)"
        )

    def test_sets_train_mode(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4, dropout_p=0.5)
        model.eval()  # start in eval mode
        loader = _tiny_classification_loader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        train_one_epoch(model, loader, criterion, optimizer, "cpu")
        assert model.training is True, (
            "train_one_epoch must call model.train() so dropout / batchnorm "
            "behave correctly during training"
        )

    def test_loss_decreases_over_epochs(self) -> None:
        torch.manual_seed(0)
        model = MLP(input_dim=8, hidden_dims=[32], output_dim=4)
        loader = _tiny_classification_loader(n=128, seed=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        first = train_one_epoch(model, loader, criterion, optimizer, "cpu")
        # Several more epochs
        last = first
        for _ in range(8):
            last = train_one_epoch(model, loader, criterion, optimizer, "cpu")
        assert last < first, (
            f"Loss should decrease across epochs. Started at {first:.4f}, ended at {last:.4f}"
        )


@pytest.mark.integration
class TestValidate:
    """validate function. spec.pdf section 5."""

    def test_returns_loss_and_accuracy(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        loader = _tiny_classification_loader()
        criterion = nn.CrossEntropyLoss()
        result = validate(model, loader, criterion, "cpu")
        assert isinstance(result, tuple) and len(result) == 2, (
            "validate must return a (loss, accuracy) tuple"
        )
        loss, acc = result
        assert isinstance(loss, float), "loss must be a Python float"
        assert isinstance(acc, float), "accuracy must be a Python float"

    def test_accuracy_in_unit_interval(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        loader = _tiny_classification_loader()
        criterion = nn.CrossEntropyLoss()
        _, acc = validate(model, loader, criterion, "cpu")
        assert 0.0 <= acc <= 1.0, f"Accuracy must be in [0, 1], got {acc}"

    def test_sets_eval_mode(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4, dropout_p=0.5)
        model.train()  # start in train mode
        loader = _tiny_classification_loader()
        criterion = nn.CrossEntropyLoss()
        validate(model, loader, criterion, "cpu")
        assert model.training is False, (
            "validate must call model.eval() to disable dropout / use BN running stats"
        )

    def test_no_parameter_updates(self) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        loader = _tiny_classification_loader()
        criterion = nn.CrossEntropyLoss()
        before = _params_snapshot(model)
        validate(model, loader, criterion, "cpu")
        after = _params_snapshot(model)
        assert not _params_changed(before, after), (
            "validate must not modify model parameters (use torch.no_grad())"
        )

    def test_perfect_predictions(self) -> None:
        """A trivial model that always predicts class 0 has measurable accuracy."""
        torch.manual_seed(0)
        # Build a balanced 4-class dataset; chance accuracy is 0.25.
        X = torch.randn(80, 8)
        y = torch.arange(80) % 4
        loader = DataLoader(TensorDataset(X, y), batch_size=20, shuffle=False)
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        criterion = nn.CrossEntropyLoss()
        _, acc = validate(model, loader, criterion, "cpu")
        # We can't assert exactness on a random model but accuracy should be sensible.
        assert 0.0 <= acc <= 1.0


@pytest.mark.integration
class TestTrainOrchestration:
    """train() end-to-end. spec.pdf section 6."""

    def _setup(self, dropout_p: float = 0.0, use_batchnorm: bool = False, seed: int = 0):
        torch.manual_seed(seed)
        model = MLP(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )
        train_loader = _tiny_classification_loader(n=64, seed=seed)
        val_loader = _tiny_classification_loader(n=32, seed=seed + 100)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        return model, train_loader, val_loader, optimizer, criterion

    def test_history_keys(self) -> None:
        model, tl, vl, opt, crit = self._setup()
        history = train(
            model=model,
            train_loader=tl,
            val_loader=vl,
            optimizer=opt,
            criterion=crit,
            max_epochs=3,
            patience=10,
            device="cpu",
        )
        required = {"train_loss", "val_loss", "val_acc", "best_epoch"}
        assert required.issubset(history.keys()), (
            f"history must contain keys {required}, got {set(history.keys())}"
        )

    def test_history_lengths(self) -> None:
        model, tl, vl, opt, crit = self._setup()
        history = train(
            model=model,
            train_loader=tl,
            val_loader=vl,
            optimizer=opt,
            criterion=crit,
            max_epochs=4,
            patience=100,  # disable early stopping
            device="cpu",
        )
        for key in ("train_loss", "val_loss", "val_acc"):
            assert len(history[key]) == 4, (
                f"With max_epochs=4 and no early stop, history['{key}'] must have length 4, "
                f"got {len(history[key])}"
            )

    def test_best_epoch_in_range(self) -> None:
        model, tl, vl, opt, crit = self._setup()
        history = train(
            model=model,
            train_loader=tl,
            val_loader=vl,
            optimizer=opt,
            criterion=crit,
            max_epochs=4,
            patience=100,
            device="cpu",
        )
        n_epochs = len(history["val_loss"])
        assert 0 <= history["best_epoch"] < n_epochs, (
            f"best_epoch must be a valid epoch index in [0, {n_epochs})"
        )

    def test_best_state_restored(self) -> None:
        """After train(), model parameters must match the best-validation snapshot."""
        torch.manual_seed(0)
        model, tl, vl, opt, crit = self._setup()
        history = train(
            model=model,
            train_loader=tl,
            val_loader=vl,
            optimizer=opt,
            criterion=crit,
            max_epochs=6,
            patience=100,
            device="cpu",
        )
        # Re-validate the model after train() returns; the val loss must equal
        # the lowest val loss seen during training (the best epoch).
        final_val_loss, _ = validate(model, vl, crit, "cpu")
        best_val_loss = min(history["val_loss"])
        assert abs(final_val_loss - best_val_loss) < 1e-3, (
            f"After train(), the model must hold the best-validation parameters. "
            f"Expected final val loss {best_val_loss:.4f}, got {final_val_loss:.4f}."
        )

    def test_early_stop_triggers(self) -> None:
        """Train on noise (no learnable signal) — early stop must fire."""
        torch.manual_seed(7)
        # Build a setup where the val loss won't improve consistently.
        train_X = torch.randn(64, 4)
        train_y = torch.randint(0, 4, (64,))  # random labels - no signal
        val_X = torch.randn(32, 4)
        val_y = torch.randint(0, 4, (32,))
        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=16, shuffle=False)
        model = MLP(input_dim=4, hidden_dims=[8], output_dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=50,
            patience=3,
            device="cpu",
        )
        assert len(history["train_loss"]) < 50, (
            f"Early stopping must trigger when val loss does not improve for "
            f"`patience` epochs. Training ran the full {len(history['train_loss'])} "
            f"epochs - check that you increment a counter on non-improvement and "
            f"break when it hits patience."
        )

    def test_scheduler_steps(self) -> None:
        """When a scheduler is provided, train() must call scheduler.step() each epoch."""
        model, tl, vl, opt, crit = self._setup()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        initial_lr = opt.param_groups[0]["lr"]
        train(
            model=model,
            train_loader=tl,
            val_loader=vl,
            optimizer=opt,
            criterion=crit,
            max_epochs=3,
            patience=100,
            device="cpu",
            scheduler=scheduler,
        )
        final_lr = opt.param_groups[0]["lr"]
        expected = initial_lr * (0.5**3)
        assert abs(final_lr - expected) < 1e-9, (
            f"With StepLR(step_size=1, gamma=0.5) over 3 epochs, lr should drop "
            f"from {initial_lr} to {expected}, got {final_lr}. "
            f"Did you call scheduler.step() once per epoch?"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis tests - 35 pts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.analysis
class TestLoadFashionMNIST:
    """load_fashion_mnist returns three DataLoaders. spec.pdf section 7.1."""

    def test_returns_three_dataloaders(self) -> None:
        result = load_fashion_mnist(batch_size=128)
        assert isinstance(result, tuple) and len(result) == 3, (
            "load_fashion_mnist must return (train, val, test) - exactly 3 loaders"
        )
        for loader in result:
            assert isinstance(loader, DataLoader), (
                f"Each return value must be a DataLoader, got {type(loader).__name__}"
            )

    def test_split_sizes(self) -> None:
        train_loader, val_loader, test_loader = load_fashion_mnist(batch_size=128)
        n_train = len(cast(Sized, train_loader.dataset))
        n_val = len(cast(Sized, val_loader.dataset))
        n_test = len(cast(Sized, test_loader.dataset))
        assert n_train + n_val + n_test == 60000, (
            f"Total examples should be 60,000 (FashionMNIST train set), "
            f"got {n_train + n_val + n_test}"
        )
        assert n_train == 48000, f"80% train split should be 48,000, got {n_train}"
        assert n_val == 6000, f"10% val split should be 6,000, got {n_val}"
        assert n_test == 6000, f"10% test split should be 6,000, got {n_test}"

    def test_batch_shape(self) -> None:
        train_loader, _, _ = load_fashion_mnist(batch_size=64)
        x, y = next(iter(train_loader))
        assert x.shape == (64, 1, 28, 28), (
            f"Each batch should have shape (B, 1, 28, 28), got {tuple(x.shape)}"
        )
        assert y.shape == (64,), f"Each batch's labels should have shape (B,), got {tuple(y.shape)}"
        assert x.dtype == torch.float32, f"Image tensors must be float32, got {x.dtype}"

    def test_labels_are_ten_classes(self) -> None:
        _, val_loader, _ = load_fashion_mnist(batch_size=128)
        all_labels = torch.cat([y for _, y in val_loader])
        assert all_labels.min().item() >= 0
        assert all_labels.max().item() <= 9, "FashionMNIST has 10 classes (labels 0-9)"


@pytest.mark.analysis
class TestBuildModel:
    """build_model returns (model, optimizer, scheduler) per config. spec.pdf section 7.2."""

    def _check_optimizer_settings(
        self,
        optimizer: torch.optim.Optimizer,
        expected_lr: float,
        expected_wd: float,
    ) -> None:
        assert isinstance(optimizer, torch.optim.SGD), (
            f"Optimizer must be SGD, got {type(optimizer).__name__}"
        )
        actual_lr = optimizer.param_groups[0]["lr"]
        actual_wd = optimizer.param_groups[0]["weight_decay"]
        assert abs(actual_lr - expected_lr) < 1e-9, f"SGD lr must be {expected_lr}, got {actual_lr}"
        assert abs(actual_wd - expected_wd) < 1e-9, (
            f"SGD weight_decay must be {expected_wd}, got {actual_wd}"
        )

    def _count(self, model: nn.Module, layer_type: type) -> int:
        return sum(1 for m in model.modules() if isinstance(m, layer_type))

    def test_returns_three_tuple(self) -> None:
        result = build_model("baseline")
        assert isinstance(result, tuple) and len(result) == 3, (
            "build_model must return (model, optimizer, scheduler) - exactly 3 items"
        )

    def test_baseline_config(self) -> None:
        model, optimizer, scheduler = build_model("baseline")
        assert isinstance(model, MLP), "baseline must build an MLP"
        assert self._count(model, Dropout) == 0, "baseline must have no Dropout layers"
        assert self._count(model, nn.BatchNorm1d) == 0, "baseline must have no BatchNorm1d layers"
        self._check_optimizer_settings(optimizer, expected_lr=0.1, expected_wd=0.0)
        assert scheduler is None, "baseline must use no LR scheduler"

    def test_dropout_config(self) -> None:
        model, optimizer, scheduler = build_model("dropout")
        assert self._count(model, Dropout) == 2, (
            "'dropout' config must have one Dropout per hidden layer (2 total)"
        )
        for d in model.modules():
            if isinstance(d, Dropout):
                assert abs(d.p - 0.3) < TOL, f"'dropout' config must use p=0.3, got p={d.p}"
        assert self._count(model, nn.BatchNorm1d) == 0, "'dropout' config must not use BatchNorm"
        self._check_optimizer_settings(optimizer, expected_lr=0.1, expected_wd=0.0)
        assert scheduler is None

    def test_batchnorm_config(self) -> None:
        model, optimizer, scheduler = build_model("batchnorm")
        assert self._count(model, nn.BatchNorm1d) == 2, (
            "'batchnorm' config must have one BatchNorm1d per hidden layer (2 total)"
        )
        assert self._count(model, Dropout) == 0, "'batchnorm' config must not use Dropout"
        self._check_optimizer_settings(optimizer, expected_lr=0.1, expected_wd=0.0)
        assert scheduler is None

    def test_both_schedule_config(self) -> None:
        model, optimizer, scheduler = build_model("both_schedule")
        assert self._count(model, Dropout) == 2, (
            "'both_schedule' must include dropout (2 Dropout layers)"
        )
        assert self._count(model, nn.BatchNorm1d) == 2, (
            "'both_schedule' must include batchnorm (2 BatchNorm1d layers)"
        )
        self._check_optimizer_settings(optimizer, expected_lr=0.1, expected_wd=1e-4)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR), (
            f"'both_schedule' must use a StepLR scheduler, got "
            f"{type(scheduler).__name__ if scheduler else 'None'}"
        )
        # StepLR stores step_size and gamma as attributes; both are guaranteed by PyTorch
        assert scheduler.step_size == 10, f"StepLR step_size must be 10, got {scheduler.step_size}"
        assert abs(scheduler.gamma - 0.5) < 1e-9, f"StepLR gamma must be 0.5, got {scheduler.gamma}"

    def test_model_has_correct_architecture(self) -> None:
        """The MLP returned by build_model uses INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM."""
        for cfg in CONFIGS:
            model, _, _ = build_model(cfg)
            linears = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
            assert len(linears) == len(HIDDEN_DIMS) + 1, (
                f"Config {cfg!r}: expected {len(HIDDEN_DIMS) + 1} Linear layers, got {len(linears)}"
            )
            in_dims = sorted([(layer.in_features, layer.out_features) for layer in linears])
            expected_dims = []
            prev = INPUT_DIM
            for h in HIDDEN_DIMS:
                expected_dims.append((prev, h))
                prev = h
            expected_dims.append((prev, OUTPUT_DIM))
            assert in_dims == sorted(expected_dims), (
                f"Config {cfg!r}: Linear layer dimensions mismatch"
            )
