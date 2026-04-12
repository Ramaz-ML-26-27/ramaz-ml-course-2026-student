from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

# Part 1 function names that must not use PyTorch.
_SCRATCH_FN_NAMES: list[str] = [
    "vector_add",
    "scalar_multiply",
    "dot_product",
    "vector_magnitude",
    "normalize_vector",
    "matrix_add",
    "matrix_vector_multiply",
    "matrix_multiply",
    "matrix_transpose",
]


def _fn_uses_torch(fn: object) -> bool:
    """Return True if the function's AST contains any 'torch.*' attribute access."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))  # type: ignore[arg-type]
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
        ):
            return True
    return False


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Set deterministic seeds for reproducible PyTorch results."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True


@pytest.fixture(autouse=True)
def check_no_torch_in_scratch(request: pytest.FixtureRequest) -> None:
    """Fail if any Part 1 (scratch) function uses torch internally."""
    if "scratch" not in request.node.keywords:
        return
    import linear_algebra as la

    for fn_name in _SCRATCH_FN_NAMES:
        fn = getattr(la, fn_name, None)
        if fn is not None and _fn_uses_torch(fn):
            pytest.fail(
                f"Part 1 function '{fn_name}' uses 'torch'. "
                "Part 1 must be implemented using only Python built-ins and the math module — "
                "no torch, no numpy."
            )
