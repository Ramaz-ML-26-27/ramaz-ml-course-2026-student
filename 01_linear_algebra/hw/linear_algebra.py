"""HW01 — Linear Algebra

Part 1: Pure-Python implementations (no torch, no numpy).
  Vectors are list[float]; matrices are list[list[float]] (row-major order).

Part 2: PyTorch implementations.
  Same math, but expressed with torch operations — no manual loops.

Run tests:   uv run pytest
Check score: uv run python score.py
"""

from __future__ import annotations

import math  # noqa: F401 — students use this in vector_magnitude

import torch

# ── Part 1: From Scratch ──────────────────────────────────────────────────────


def vector_add(u: list[float], v: list[float]) -> list[float]:
    """Return the element-wise sum of two vectors.

    Both vectors must have the same length.

    Args:
        u: First vector.
        v: Second vector, same length as u.

    Returns:
        A new vector w where w[i] = u[i] + v[i].

    Example:
        >>> vector_add([1.0, 2.0], [3.0, 4.0])
        [4.0, 6.0]

    Hint:
        Think about how to iterate over two lists simultaneously, pairing
        their elements at each position.
    """
    raise NotImplementedError("Implement vector_add()")


def scalar_multiply(c: float, v: list[float]) -> list[float]:
    """Scale every element of a vector by a scalar.

    Args:
        c: The scalar multiplier.
        v: The vector to scale.

    Returns:
        A new vector w where w[i] = c * v[i].

    Example:
        >>> scalar_multiply(3.0, [1.0, 2.0, 3.0])
        [3.0, 6.0, 9.0]

    Hint:
        You know how to visit every element in a list. What would you do
        to each one?
    """
    raise NotImplementedError("Implement scalar_multiply()")


def dot_product(u: list[float], v: list[float]) -> float:
    """Compute the dot product (inner product) of two vectors.

    The dot product is the sum of element-wise products:
        dot(u, v) = u[0]*v[0] + u[1]*v[1] + ... + u[n-1]*v[n-1]

    Args:
        u: First vector.
        v: Second vector, same length as u.

    Returns:
        A single float — the dot product of u and v.

    Example:
        >>> dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        32.0

    Hint:
        You already know how to pair elements from two lists. The dot product
        needs one more step: combine those products into a single number.
    """
    raise NotImplementedError("Implement dot_product()")


def vector_magnitude(v: list[float]) -> float:
    """Compute the Euclidean (L2) magnitude (length) of a vector.

    magnitude(v) = sqrt(v[0]^2 + v[1]^2 + ... + v[n-1]^2)

    Args:
        v: The input vector.

    Returns:
        A non-negative float — the magnitude of v.

    Example:
        >>> vector_magnitude([3.0, 4.0])
        5.0

    Hint:
        Look at the formula in the docstring — it expresses magnitude in terms
        of an operation you've already implemented.
    """
    raise NotImplementedError("Implement vector_magnitude()")


def normalize_vector(v: list[float]) -> list[float]:
    """Return the unit vector in the same direction as v.

    A unit vector has magnitude 1. To normalize, divide each element by the
    magnitude of v.

    Args:
        v: The vector to normalize.

    Returns:
        A new vector with magnitude 1, pointing in the same direction as v.

    Raises:
        ValueError: If v is the zero vector (magnitude == 0).

    Example:
        >>> normalize_vector([3.0, 4.0])
        [0.6, 0.8]

    Hint:
        You have all the pieces from earlier in Part 1. Think about what needs
        to be true about the magnitude before dividing, and what should happen
        if that condition fails.
    """
    raise NotImplementedError("Implement normalize_vector()")


def matrix_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Return the element-wise sum of two matrices.

    Both matrices must have identical dimensions (same number of rows and
    the same number of columns in each row).

    Args:
        A: First matrix (list of rows).
        B: Second matrix, same shape as A.

    Returns:
        A new matrix C where C[i][j] = A[i][j] + B[i][j].

    Example:
        >>> matrix_add([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        [[6.0, 8.0], [10.0, 12.0]]

    Hint:
        You have a function that adds two vectors. How could you apply it
        to each pair of corresponding rows?
    """
    raise NotImplementedError("Implement matrix_add()")


def matrix_vector_multiply(A: list[list[float]], v: list[float]) -> list[float]:
    """Multiply a matrix A by a column vector v.

    The i-th element of the output is the dot product of the i-th row of A
    with v:
        result[i] = dot(A[i], v)

    Args:
        A: A matrix with shape (m, n) — m rows, n columns.
        v: A vector of length n.

    Returns:
        A vector of length m.

    Example:
        >>> matrix_vector_multiply([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
        [3.0, 7.0]

    Hint:
        The docstring gives you the mathematical rule. You have a function that
        computes exactly what each output element requires — apply it across the
        rows.
    """
    raise NotImplementedError("Implement matrix_vector_multiply()")


def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices A and B.

    If A has shape (m, k) and B has shape (k, n), the result has shape (m, n).
    The (i, j) entry of the result is:
        C[i][j] = dot product of row i of A with column j of B

    Args:
        A: Matrix with shape (m, k).
        B: Matrix with shape (k, n).

    Returns:
        Matrix C with shape (m, n).

    Example:
        >>> matrix_multiply([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        [[19.0, 22.0], [43.0, 50.0]]

    Hint:
        The hard part is accessing B's columns. Think about what operation
        could turn columns into something easier to work with — you've already
        implemented it.
    """
    raise NotImplementedError("Implement matrix_multiply()")


def matrix_transpose(A: list[list[float]]) -> list[list[float]]:
    """Return the transpose of matrix A.

    The transpose swaps rows and columns: T[i][j] = A[j][i].

    Args:
        A: A matrix with shape (m, n).

    Returns:
        The transposed matrix with shape (n, m).

    Example:
        >>> matrix_transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    Hint:
        Python's `zip` function can be surprisingly useful here. Think about
        what happens when you pass a list of lists as separate arguments — try
        it on a small example in the REPL.
    """
    raise NotImplementedError("Implement matrix_transpose()")


# ── Part 2: PyTorch Tensors ───────────────────────────────────────────────────


def tensor_dot_product(u: torch.Tensor, v: torch.Tensor) -> float:
    """Compute the dot product of two 1-D tensors.

    Args:
        u: 1-D tensor.
        v: 1-D tensor, same length as u.

    Returns:
        A Python float — the dot product.

    Example:
        >>> tensor_dot_product(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        11.0

    Hint:
        The reference card shows both the relevant function and the conversion
        step needed to return a Python float.
    """
    raise NotImplementedError("Implement tensor_dot_product()")


def tensor_magnitude(v: torch.Tensor) -> float:
    """Compute the Euclidean (L2) norm of a tensor.

    Args:
        v: 1-D tensor.

    Returns:
        A Python float — the L2 norm of v.

    Example:
        >>> tensor_magnitude(torch.tensor([3.0, 4.0]))
        5.0

    Hint:
        Check the PyTorch cheat sheet in the reference card — the return type
        matters here.
    """
    raise NotImplementedError("Implement tensor_magnitude()")


def tensor_normalize(v: torch.Tensor) -> torch.Tensor:
    """Return the unit vector in the same direction as v.

    Args:
        v: 1-D tensor.

    Returns:
        A tensor with magnitude 1 pointing in the same direction as v.

    Raises:
        ValueError: If v is the zero vector.

    Example:
        >>> tensor_normalize(torch.tensor([3.0, 4.0]))
        tensor([0.6000, 0.8000])

    Hint:
        You've implemented normalization in Part 1 — the logic is the same.
        Think about what PyTorch gives you when you divide a tensor by a scalar
        tensor.
    """
    raise NotImplementedError("Implement tensor_normalize()")


def tensor_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Multiply two 2-D tensors (matrices) using the @ operator.

    Args:
        A: 2-D tensor with shape (m, k).
        B: 2-D tensor with shape (k, n).

    Returns:
        2-D tensor with shape (m, n) — the matrix product.

    Example:
        >>> A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> tensor_matmul(A, B)
        tensor([[19., 22.],
                [43., 50.]])

    Hint:
        Check the reference card for how PyTorch handles matrix
        multiplication.
    """
    raise NotImplementedError("Implement tensor_matmul()")


def tensor_transpose(A: torch.Tensor) -> torch.Tensor:
    """Return the transpose of a 2-D (or batched) tensor.

    Args:
        A: A tensor with at least 2 dimensions.

    Returns:
        The transposed tensor. For a 2-D matrix of shape (m, n), this has
        shape (n, m).

    Example:
        >>> A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> tensor_transpose(A)
        tensor([[1., 4.],
                [2., 5.],
                [3., 6.]])

    Hint:
        Check the reference card — PyTorch tensors have a property for
        transposing that works correctly for both 2-D and batched tensors.
    """
    raise NotImplementedError("Implement tensor_transpose()")


def column_means(A: torch.Tensor) -> torch.Tensor:
    """Compute the mean of each column of a matrix.

    For a matrix with m rows and n columns, this returns a vector of
    length n where entry j is the average of all values in column j.
    This is equivalent to computing the mean across rows (dim=0).

    Args:
        A: A 2-D tensor of shape (m, n).

    Returns:
        A 1-D tensor of shape (n,) — the column-wise means.

    Example:
        >>> A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> column_means(A)
        tensor([3., 4.])  # (1+3+5)/3=3, (2+4+6)/3=4

    Hint:
        Think about which dimension to collapse to get one value per column.
        Check the PyTorch cheat sheet for how dim values work.
    """
    raise NotImplementedError("Implement column_means()")


def row_normalize(A: torch.Tensor) -> torch.Tensor:
    """Normalize each row of a matrix to unit length.

    For a matrix with m rows, this produces a new matrix where every row
    has Euclidean norm equal to 1. This is useful for comparing rows as
    directions regardless of their magnitude — e.g., normalizing data
    samples before computing cosine similarities.

    Args:
        A: A 2-D tensor of shape (m, n).

    Returns:
        A 2-D tensor of shape (m, n) where each row has norm 1.

    Raises:
        ValueError: If any row of A is the zero vector (cannot be normalized).

    Example:
        >>> A = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        >>> row_normalize(A)
        tensor([[0.6000, 0.8000],
                [0.0000, 1.0000]])

    Hint:
        Compute norms along the row dimension. When you try to divide A by the
        result, pay attention to whether the shapes are compatible for
        broadcasting — you may need to preserve a dimension.
    """
    raise NotImplementedError("Implement row_normalize()")


def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> float:
    """Compute the cosine similarity between two 1-D tensors.

    Cosine similarity measures the angle between two vectors:

        cos_sim(u, v) = (u . v) / (||u|| * ||v||)

    It ranges from -1 (opposite directions) to 1 (same direction).
    A value of 0 means the vectors are perpendicular.

    Args:
        u: 1-D tensor.
        v: 1-D tensor, same length as u.

    Returns:
        A Python float in [-1, 1].

    Raises:
        ValueError: If either u or v is the zero vector.

    Example:
        >>> cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        1.0
        >>> cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
        0.0  # perpendicular

    Hint:
        The formula is in the docstring. You have all the building blocks
        from earlier in Part 2 — think about what each component of the
        formula maps to in PyTorch. Don't forget to handle the zero-vector
        case.
    """
    raise NotImplementedError("Implement cosine_similarity()")


def gram_matrix(A: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of A: G = A^T A.

    The Gram matrix is always square (n x n if A has n columns) and
    symmetric. Its (i, j) entry is the dot product of column i of A
    with column j of A.

    Gram matrices appear throughout ML: in linear regression (the normal
    equations involve X^T X), in kernel methods, and in style transfer.
    You will use this exact operation again in HW03 when implementing
    least-squares regression.

    Args:
        A: A 2-D tensor of shape (m, n).

    Returns:
        A 2-D tensor of shape (n, n).

    Example:
        >>> A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> gram_matrix(A)
        tensor([[35., 44.],
                [44., 56.]])

    Hint:
        You need the transpose of A and a matrix multiply. Think carefully
        about the order: is it A @ A.mT or A.mT @ A? Check the output shape.
    """
    raise NotImplementedError("Implement gram_matrix()")


def solve_linear_system(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve the linear system Ax = b for the unknown vector x.

    Given a square invertible matrix A and a right-hand-side vector b,
    find x such that A @ x == b.

    This is the tensor equivalent of solving a system of linear equations:
        a_00*x_0 + a_01*x_1 + ... = b_0
        a_10*x_0 + a_11*x_1 + ... = b_1
        ...

    torch.linalg.solve(A, b) computes this efficiently using LU decomposition
    — it is numerically more stable than computing the inverse of A directly
    (i.e., do NOT use A.inverse() @ b).

    Args:
        A: Square matrix of shape (n, n). Must be non-singular (invertible).
        b: Right-hand-side vector of shape (n,) or matrix of shape (n, k).

    Returns:
        Solution tensor x such that A @ x is close to b.

    Example:
        >>> A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        >>> b = torch.tensor([5.0, 10.0])
        >>> x = solve_linear_system(A, b)
        >>> print(x)
        tensor([1., 3.])  # 2(1) + 3 = 5 ✓ and 1 + 3(3) = 10 ✓

    Hint:
        The docstring tells you the right function to use and why a more
        obvious approach is discouraged. Read it carefully.
    """
    raise NotImplementedError("Implement solve_linear_system()")
