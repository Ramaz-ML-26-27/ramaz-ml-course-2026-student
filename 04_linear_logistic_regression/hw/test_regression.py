"""Tests for HW03 - regression.py and analysis.py.

Tests are organized in three layers (see spec.pdf section 5):

    @pytest.mark.unit         — one method at a time (40 pts total)
    @pytest.mark.integration  — full fit/predict end-to-end (30 pts total)
    @pytest.mark.analysis     — analysis.py helper functions (20 pts total)

Each test class corresponds to one scoring bucket. All tests in a class must
pass to earn the bucket's credit.

Run with: uv run pytest
"""

from __future__ import annotations

import math

import pytest
import torch

from analysis import (
    load_classification_data,
    load_regression_data,
    train_linear_model,
    train_logistic_model,
)
from regression import LinearRegression, LogisticRegression, sigmoid

TOL = 1e-4


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_linear_dataset(
    n: int = 200, d: int = 4, noise: float = 0.05, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic dataset where y = X @ w_true + b_true + small noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w_true = torch.randn(d, generator=g)
    b_true = torch.randn(1, generator=g)[0]
    y = X @ w_true + b_true + noise * torch.randn(n, generator=g)
    return X, y, w_true, b_true


def _make_logistic_dataset(
    n: int = 200, d: int = 3, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic linearly separable binary dataset."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w_true = torch.randn(d, generator=g)
    b_true = torch.randn(1, generator=g)[0]
    logits = X @ w_true + b_true
    y = (logits > 0).float()
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — 40 pts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSigmoid:
    """sigmoid() free function. spec.pdf §2."""

    def test_at_zero(self) -> None:
        assert torch.allclose(sigmoid(torch.tensor(0.0)), torch.tensor(0.5), atol=TOL), (
            "sigmoid(0) must be 0.5"
        )

    def test_large_positive(self) -> None:
        assert sigmoid(torch.tensor(10.0)).item() > 0.99, "sigmoid(10) should be close to 1"

    def test_large_negative(self) -> None:
        assert sigmoid(torch.tensor(-10.0)).item() < 0.01, "sigmoid(-10) should be close to 0"

    def test_output_in_zero_one(self) -> None:
        z = torch.linspace(-5.0, 5.0, 20)
        out = sigmoid(z)
        assert (out > 0).all() and (out < 1).all(), "All sigmoid outputs must be strictly in (0, 1)"

    def test_symmetry(self) -> None:
        z = torch.tensor([1.0, -2.0, 0.5, 3.0])
        assert torch.allclose(sigmoid(-z), 1.0 - sigmoid(z), atol=TOL), (
            "sigmoid(-z) must equal 1 - sigmoid(z)"
        )

    def test_shape_preserved(self) -> None:
        z = torch.randn(4, 3)
        assert sigmoid(z).shape == z.shape, "sigmoid must preserve input shape"


@pytest.mark.unit
class TestLinearInit:
    """LinearRegression.__init__. spec.pdf §3."""

    def test_w_shape_and_zeros(self) -> None:
        m = LinearRegression(n_features=5)
        assert m.w.shape == (5,), f"w must have shape (5,), got {m.w.shape}"
        assert torch.allclose(m.w, torch.zeros(5), atol=TOL), "w must initialize to zeros"

    def test_b_scalar_zero(self) -> None:
        m = LinearRegression(n_features=3)
        assert m.b.dim() == 0, f"b must be a 0-dim scalar tensor, got dim={m.b.dim()}"
        assert abs(m.b.item()) < TOL, "b must initialize to zero"

    def test_lr_stored(self) -> None:
        m = LinearRegression(n_features=2, lr=0.123)
        assert m.lr == 0.123, "lr must be stored as self.lr"

    def test_default_lr(self) -> None:
        m = LinearRegression(n_features=2)
        assert m.lr == 0.01, "default lr should be 0.01 per spec"


@pytest.mark.unit
class TestLinearForward:
    """LinearRegression.forward(X). spec.pdf §3."""

    def test_basic(self) -> None:
        m = LinearRegression(n_features=3)
        m.w = torch.tensor([1.0, 1.0, 1.0])
        m.b = torch.tensor(0.5)
        X = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(m.forward(X), torch.tensor([6.5]), atol=TOL), (
            "forward([[1,2,3]]) with w=[1,1,1], b=0.5 should be [6.5]"
        )

    def test_batch(self) -> None:
        m = LinearRegression(n_features=2)
        m.w = torch.tensor([2.0, -1.0])
        m.b = torch.tensor(1.0)
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        assert torch.allclose(m.forward(X), torch.tensor([3.0, 0.0, 2.0]), atol=TOL), (
            "forward must handle batches"
        )

    def test_output_shape(self) -> None:
        m = LinearRegression(n_features=5)
        m.w = torch.randn(5)
        X = torch.randn(7, 5)
        assert m.forward(X).shape == (7,), "forward output must have shape (n,)"

    def test_zero_weights(self) -> None:
        m = LinearRegression(n_features=3)
        m.b = torch.tensor(5.0)
        X = torch.randn(4, 3)
        assert torch.allclose(m.forward(X), torch.full((4,), 5.0), atol=TOL), (
            "with w=0, all predictions should equal b"
        )


@pytest.mark.unit
class TestLinearLoss:
    """LinearRegression.loss (MSE). spec.pdf §3."""

    def test_perfect(self) -> None:
        m = LinearRegression(n_features=2)
        y = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(m.loss(y, y), torch.tensor(0.0), atol=TOL), (
            "MSE of perfect predictions must be 0"
        )

    def test_known_value(self) -> None:
        m = LinearRegression(n_features=2)
        y_pred = torch.tensor([0.0, 0.0, 0.0])
        y_true = torch.tensor([1.0, 2.0, 3.0])
        # squared errors: 1, 4, 9 → mean: 14/3
        assert torch.allclose(m.loss(y_pred, y_true), torch.tensor(14.0 / 3.0), atol=TOL), (
            "MSE([0,0,0], [1,2,3]) should be 14/3"
        )

    def test_scalar(self) -> None:
        m = LinearRegression(n_features=2)
        y_pred = torch.randn(10)
        y_true = torch.randn(10)
        assert m.loss(y_pred, y_true).dim() == 0, "MSE must return a scalar tensor"

    def test_non_negative(self) -> None:
        m = LinearRegression(n_features=2)
        for _ in range(5):
            y_pred = torch.randn(8)
            y_true = torch.randn(8)
            assert m.loss(y_pred, y_true).item() >= 0, "MSE must be non-negative"


@pytest.mark.unit
class TestLinearGradient:
    """LinearRegression.gradient. spec.pdf §3."""

    def test_zero_at_minimum(self) -> None:
        m = LinearRegression(n_features=2)
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = torch.tensor([2.0, 4.0, 6.0])
        dw, db = m.gradient(X, y, y.clone())
        assert torch.allclose(dw, torch.zeros(2), atol=TOL), "dw must be 0 at the minimum"
        assert abs(db.item()) < TOL, "db must be 0 at the minimum"

    def test_shapes(self) -> None:
        m = LinearRegression(n_features=3)
        X = torch.randn(5, 3)
        y_true = torch.randn(5)
        y_pred = torch.randn(5)
        dw, db = m.gradient(X, y_true, y_pred)
        assert dw.shape == (3,), f"dw must have shape (3,), got {dw.shape}"
        assert db.dim() == 0, f"db must be a scalar tensor, got dim={db.dim()}"

    def test_matches_formula(self) -> None:
        m = LinearRegression(n_features=2)
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_true = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([3.0, 5.0])
        dw, db = m.gradient(X, y_true, y_pred)
        n = X.shape[0]
        residuals = y_pred - y_true
        expected_dw = (2.0 / n) * X.T @ residuals
        expected_db = (2.0 / n) * residuals.sum()
        assert torch.allclose(dw, expected_dw, atol=TOL), (
            "dw must equal (2/n)·X^T·(y_pred − y_true)"
        )
        assert torch.allclose(db, expected_db, atol=TOL), "db must equal (2/n)·sum(y_pred − y_true)"

    def test_direction(self) -> None:
        m = LinearRegression(n_features=1)
        X = torch.tensor([[1.0], [2.0], [3.0]])
        y_true = torch.tensor([0.0, 0.0, 0.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        dw, db = m.gradient(X, y_true, y_pred)
        assert dw[0].item() > 0 and db.item() > 0, (
            "Over-predicting with positive features should give positive gradients"
        )


@pytest.mark.unit
class TestLogisticInit:
    """LogisticRegression.__init__. spec.pdf §4."""

    def test_w_zeros(self) -> None:
        m = LogisticRegression(n_features=4)
        assert m.w.shape == (4,) and torch.allclose(m.w, torch.zeros(4), atol=TOL), (
            "w must initialize to zero vector of shape (n_features,)"
        )

    def test_b_scalar_zero(self) -> None:
        m = LogisticRegression(n_features=3)
        assert m.b.dim() == 0 and abs(m.b.item()) < TOL, "b must initialize to a scalar 0"

    def test_default_lr(self) -> None:
        m = LogisticRegression(n_features=2)
        assert m.lr == 0.5, "default lr should be 0.5 per spec"


@pytest.mark.unit
class TestLogisticForward:
    """LogisticRegression.forward. spec.pdf §4."""

    def test_zero_params_gives_half(self) -> None:
        m = LogisticRegression(n_features=3)
        X = torch.randn(5, 3)
        out = m.forward(X)
        assert torch.allclose(out, torch.full((5,), 0.5), atol=TOL), (
            "with w=0, b=0, all outputs should be 0.5"
        )

    def test_in_zero_one(self) -> None:
        m = LogisticRegression(n_features=4)
        m.w = torch.randn(4)
        m.b = torch.randn(1)[0]
        X = torch.randn(10, 4)
        out = m.forward(X)
        assert (out > 0).all() and (out < 1).all(), (
            "logistic forward outputs must be strictly in (0, 1)"
        )

    def test_shape(self) -> None:
        m = LogisticRegression(n_features=3)
        m.w = torch.randn(3)
        X = torch.randn(8, 3)
        assert m.forward(X).shape == (8,), "forward output must have shape (n,)"

    def test_forward_equals_sigmoid_of_linear_combination(self) -> None:
        m = LogisticRegression(n_features=2)
        m.w = torch.randn(2)
        m.b = torch.randn(1)[0]
        X = torch.randn(4, 2)
        assert torch.allclose(m.forward(X), sigmoid(X @ m.w + m.b), atol=TOL), (
            "forward(X) must equal sigmoid(X @ w + b)"
        )


@pytest.mark.unit
class TestLogisticLoss:
    """LogisticRegression.loss (BCE). spec.pdf §4."""

    def test_near_perfect(self) -> None:
        m = LogisticRegression(n_features=2)
        y_pred = torch.tensor([0.99, 0.01, 0.99])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        assert m.loss(y_pred, y_true).item() < 0.02, (
            "near-perfect predictions should give small BCE"
        )

    def test_scalar(self) -> None:
        m = LogisticRegression(n_features=2)
        y_pred = torch.tensor([0.5, 0.5, 0.5])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        assert m.loss(y_pred, y_true).dim() == 0, "BCE must return a scalar tensor"

    def test_non_negative(self) -> None:
        m = LogisticRegression(n_features=2)
        y_pred = torch.tensor([0.3, 0.7, 0.5, 0.9])
        y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
        assert m.loss(y_pred, y_true).item() >= 0, "BCE must be non-negative"

    def test_log_two_at_half(self) -> None:
        m = LogisticRegression(n_features=2)
        y_pred = torch.tensor([0.5, 0.5])
        y_true = torch.tensor([0.0, 1.0])
        assert abs(m.loss(y_pred, y_true).item() - math.log(2)) < 1e-3, (
            "BCE with all predictions = 0.5 must equal log(2)"
        )

    def test_confident_wrong_penalized(self) -> None:
        m = LogisticRegression(n_features=2)
        y_true = torch.tensor([1.0])
        wrong = m.loss(torch.tensor([0.01]), y_true).item()
        unsure = m.loss(torch.tensor([0.5]), y_true).item()
        assert wrong > unsure, "confident wrong prediction should give higher loss than uncertain"


@pytest.mark.unit
class TestLogisticGradient:
    """LogisticRegression.gradient. spec.pdf §4."""

    def test_zero_at_residual_zero(self) -> None:
        m = LogisticRegression(n_features=2)
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([0.0, 1.0])
        dw, db = m.gradient(X, y, y.clone())
        assert torch.allclose(dw, torch.zeros(2), atol=TOL), "dw must be 0 when residuals are 0"
        assert abs(db.item()) < TOL, "db must be 0 when residuals are 0"

    def test_shapes(self) -> None:
        m = LogisticRegression(n_features=4)
        X = torch.randn(6, 4)
        y_true = torch.randint(0, 2, (6,)).float()
        y_pred = sigmoid(torch.randn(6))
        dw, db = m.gradient(X, y_true, y_pred)
        assert dw.shape == (4,), f"dw must have shape (4,), got {dw.shape}"
        assert db.dim() == 0, f"db must be scalar, got dim={db.dim()}"

    def test_matches_formula(self) -> None:
        m = LogisticRegression(n_features=2)
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_true = torch.tensor([0.0, 1.0, 0.0])
        y_pred = torch.tensor([0.6, 0.4, 0.7])
        dw, db = m.gradient(X, y_true, y_pred)
        n = X.shape[0]
        residuals = y_pred - y_true
        expected_dw = (1.0 / n) * X.T @ residuals
        expected_db = (1.0 / n) * residuals.sum()
        assert torch.allclose(dw, expected_dw, atol=TOL), (
            "dw must equal (1/n)·X^T·(y_pred − y_true) — note 1/n, not 2/n"
        )
        assert torch.allclose(db, expected_db, atol=TOL), "db must equal (1/n)·sum(y_pred − y_true)"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — 30 pts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestLinearFit:
    """LinearRegression.fit end-to-end. spec.pdf §3."""

    def test_loss_history_shape(self) -> None:
        X, y, _, _ = _make_linear_dataset()
        m = LinearRegression(n_features=X.shape[1], lr=0.05)
        history = m.fit(X, y, epochs=50)
        assert isinstance(history, list), "fit must return a list"
        assert len(history) == 50, "history must have one entry per epoch"
        assert all(isinstance(v, float) for v in history), (
            "each history entry must be a Python float, not a tensor"
        )

    def test_loss_decreases(self) -> None:
        X, y, _, _ = _make_linear_dataset()
        m = LinearRegression(n_features=X.shape[1], lr=0.05)
        history = m.fit(X, y, epochs=100)
        assert history[-1] < history[0], "loss must decrease over training"
        assert history[-1] < 0.5 * history[0], (
            "loss should drop substantially over 100 epochs at lr=0.05"
        )

    def test_recovers_coefficients(self) -> None:
        X, y, w_true, b_true = _make_linear_dataset(n=400, d=3, noise=0.01)
        m = LinearRegression(n_features=3, lr=0.05)
        m.fit(X, y, epochs=500)
        assert torch.allclose(m.w, w_true, atol=0.1), (
            f"fit must recover w_true within 0.1. Got {m.w.tolist()} vs {w_true.tolist()}"
        )
        assert abs(m.b.item() - b_true.item()) < 0.1, (
            f"fit must recover b_true within 0.1. Got {m.b.item()} vs {b_true.item()}"
        )


@pytest.mark.integration
class TestLinearPredict:
    """LinearRegression.predict. spec.pdf §3."""

    def test_predict_equals_forward(self) -> None:
        X, y, _, _ = _make_linear_dataset()
        m = LinearRegression(n_features=X.shape[1], lr=0.05)
        m.fit(X, y, epochs=10)
        X_new = torch.randn(8, X.shape[1])
        assert torch.allclose(m.predict(X_new), m.forward(X_new), atol=TOL), (
            "for linear regression, predict must equal forward"
        )

    def test_predict_shape(self) -> None:
        m = LinearRegression(n_features=4)
        X_new = torch.randn(11, 4)
        assert m.predict(X_new).shape == (11,), "predict output must have shape (n,)"


@pytest.mark.integration
class TestLogisticFit:
    """LogisticRegression.fit end-to-end. spec.pdf §4."""

    def test_loss_history_shape(self) -> None:
        X, y = _make_logistic_dataset()
        m = LogisticRegression(n_features=X.shape[1], lr=0.5)
        history = m.fit(X, y, epochs=50)
        assert isinstance(history, list) and len(history) == 50
        assert all(isinstance(v, float) for v in history), (
            "each history entry must be a Python float"
        )

    def test_loss_decreases(self) -> None:
        X, y = _make_logistic_dataset()
        m = LogisticRegression(n_features=X.shape[1], lr=0.5)
        history = m.fit(X, y, epochs=200)
        assert history[-1] < history[0], "loss must decrease"
        assert history[-1] < 0.5 * history[0], "loss should drop substantially"

    def test_classifies_separable(self) -> None:
        X, y = _make_logistic_dataset(n=400, d=3, seed=1)
        m = LogisticRegression(n_features=3, lr=0.5)
        m.fit(X, y, epochs=500)
        preds = m.predict(X)
        accuracy = (preds == y).float().mean().item()
        assert accuracy > 0.95, (
            f"on linearly separable data, accuracy should exceed 95%. Got {accuracy:.3f}"
        )


@pytest.mark.integration
class TestLogisticPredict:
    """LogisticRegression.predict_proba and predict. spec.pdf §4."""

    def test_proba_in_zero_one(self) -> None:
        m = LogisticRegression(n_features=3)
        m.w = torch.randn(3)
        m.b = torch.randn(1)[0]
        X = torch.randn(20, 3)
        out = m.predict_proba(X)
        assert (out > 0).all() and (out < 1).all(), "predict_proba must be in (0, 1)"

    def test_proba_equals_forward(self) -> None:
        m = LogisticRegression(n_features=3)
        m.w = torch.randn(3)
        X = torch.randn(7, 3)
        assert torch.allclose(m.predict_proba(X), m.forward(X), atol=TOL), (
            "predict_proba must equal forward"
        )

    def test_predict_threshold(self) -> None:
        m = LogisticRegression(n_features=2)
        m.w = torch.randn(2)
        m.b = torch.randn(1)[0]
        X = torch.randn(15, 2)
        proba = m.predict_proba(X)
        hard = m.predict(X)
        expected = (proba > 0.5).float()
        assert torch.allclose(hard, expected, atol=TOL), (
            "predict must threshold predict_proba at 0.5"
        )

    def test_predict_values_in_zero_one(self) -> None:
        m = LogisticRegression(n_features=2)
        X = torch.randn(20, 2)
        out = m.predict(X)
        assert torch.all((out == 0) | (out == 1)), "predict output must contain only 0 and 1"


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis tests — 20 pts (analysis.py functions)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.analysis
class TestLoadRegressionData:
    """analysis.load_regression_data."""

    def test_returns_four_tensors(self) -> None:
        result = load_regression_data()
        assert isinstance(result, tuple) and len(result) == 4
        assert all(isinstance(t, torch.Tensor) for t in result)

    def test_dtype_float32(self) -> None:
        for t in load_regression_data():
            assert t.dtype == torch.float32, f"all returned tensors must be float32, got {t.dtype}"

    def test_shapes(self) -> None:
        X_train, X_test, y_train, y_test = load_regression_data()
        assert X_train.shape[1] == 8, "California Housing has 8 features"
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[0] > X_test.shape[0], "train set should be larger than test set"

    def test_no_nans(self) -> None:
        for t in load_regression_data():
            assert not torch.isnan(t).any(), "load_regression_data returned NaN values"


@pytest.mark.analysis
class TestTrainLinearModel:
    """analysis.train_linear_model. Returns (model, loss_history)."""

    def test_returns_model_and_history(self) -> None:
        X_train, _, y_train, _ = load_regression_data()
        result = train_linear_model(X_train, y_train, lr=0.01, epochs=10)
        assert isinstance(result, tuple) and len(result) == 2, (
            "train_linear_model must return (model, loss_history)"
        )
        model, history = result
        assert isinstance(model, LinearRegression), "first return must be LinearRegression"
        assert isinstance(history, list), "second return must be a list"
        assert len(history) == 10, "history length must equal epochs"

    def test_loss_decreases(self) -> None:
        X_train, _, y_train, _ = load_regression_data()
        _, history = train_linear_model(X_train, y_train, lr=0.01, epochs=100)
        assert history[-1] < history[0], "loss must decrease"

    def test_model_w_shape(self) -> None:
        X_train, _, y_train, _ = load_regression_data()
        model, _ = train_linear_model(X_train, y_train, lr=0.01, epochs=5)
        assert model.w.shape == (X_train.shape[1],), (
            f"model.w must have shape ({X_train.shape[1]},)"
        )


@pytest.mark.analysis
class TestLoadClassificationData:
    """analysis.load_classification_data — Iris binary subset."""

    def test_returns_four_tensors(self) -> None:
        result = load_classification_data()
        assert isinstance(result, tuple) and len(result) == 4
        assert all(isinstance(t, torch.Tensor) for t in result)

    def test_dtype_float32(self) -> None:
        for t in load_classification_data():
            assert t.dtype == torch.float32

    def test_features_two(self) -> None:
        X_train, X_test, _, _ = load_classification_data()
        assert X_train.shape[1] == 2, "use only the first two Iris features"
        assert X_test.shape[1] == 2

    def test_labels_binary(self) -> None:
        _, _, y_train, y_test = load_classification_data()
        for y in (y_train, y_test):
            unique = torch.unique(y)
            assert torch.all((unique == 0) | (unique == 1)), (
                f"labels must be in {{0, 1}}, got {unique}"
            )


@pytest.mark.analysis
class TestTrainLogisticModel:
    """analysis.train_logistic_model."""

    def test_returns_model_and_history(self) -> None:
        X_train, _, y_train, _ = load_classification_data()
        result = train_logistic_model(X_train, y_train, lr=0.5, epochs=10)
        assert isinstance(result, tuple) and len(result) == 2
        model, history = result
        assert isinstance(model, LogisticRegression), "first return must be LogisticRegression"
        assert isinstance(history, list)
        assert len(history) == 10

    def test_loss_decreases(self) -> None:
        X_train, _, y_train, _ = load_classification_data()
        _, history = train_logistic_model(X_train, y_train, lr=0.5, epochs=100)
        assert history[-1] < history[0], "loss must decrease"

    def test_classifies_iris(self) -> None:
        """Setosa vs. versicolor is linearly separable; should reach high accuracy."""
        X_train, _, y_train, _ = load_classification_data()
        model, _ = train_logistic_model(X_train, y_train, lr=0.5, epochs=300)
        accuracy = (model.predict(X_train) == y_train).float().mean().item()
        assert accuracy > 0.95, (
            f"setosa vs. versicolor should be classified with >95% accuracy. Got {accuracy:.3f}"
        )


