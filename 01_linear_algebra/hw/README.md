# HW01 — Linear Algebra

In this assignment you will implement core linear algebra operations in two ways:
first from scratch in pure Python, then using PyTorch tensors.

By the end you will have built every fundamental building block — vector addition,
dot products, matrix multiplication, and solving linear systems — and you will
understand exactly what PyTorch is doing under the hood when it performs these
operations during model training.

---

## Prerequisites

- HW00 completed (comfortable with Python lists and functions)
- Attended the Linear Algebra lecture (vectors, matrices, dot products, matrix
  multiplication, linear systems)

---

## Assignment structure

### Part 1 — From Scratch (27 pts)

Implement the functions in `linear_algebra.py` using only Python built-ins and
the `math` module. No torch, no numpy. Vectors are `list[float]`; matrices are
`list[list[float]]` (stored in row-major order).

Functions to implement:
- `vector_add`
- `scalar_multiply`
- `dot_product`
- `vector_magnitude`
- `normalize_vector`
- `matrix_add`
- `matrix_vector_multiply`
- `matrix_multiply`
- `matrix_transpose`

### Part 2 — PyTorch Tensors (27 pts)

Same operations, but implemented using PyTorch. No manual loops — use PyTorch
operators and functions.

Functions to implement:
- `tensor_dot_product`
- `tensor_magnitude`
- `tensor_normalize`
- `tensor_matmul`
- `tensor_transpose`
- `column_means`
- `row_normalize`
- `cosine_similarity`
- `gram_matrix`
- `solve_linear_system`

### Math Exercises (graded separately)

A separate PDF of written math exercises accompanies this assignment. Complete
them by hand and submit as a scanned or photographed PDF to Gradescope alongside
this coding assignment.

---

## Setup

Install dependencies (only needed once):

```bash
uv sync
```

(`uv` is a fast Python package manager — if you've used `pip` before, `uv sync` is equivalent to `pip install -r requirements.txt`.)

---

## Running tests

Run the full test suite:

```bash
uv run pytest
```

Run only Part 1:

```bash
uv run pytest -m scratch
```

Run only Part 2:

```bash
uv run pytest -m pytorch
```

---

## Checking your score

```bash
uv run python score.py          # full score
uv run python score.py scratch  # Part 1 only
uv run python score.py pytorch  # Part 2 only
```

Every test class is all-or-nothing: you earn full points for a class only when
**all** tests in that class pass. The score script shows which tests are failing
so you know what to fix.

---

## Part 3 — Reflection (2 pts, manually graded)

After completing Parts 1 and 2, add answers to these questions as comments at
the **bottom** of your `linear_algebra.py` file:

1. In `column_means`, you pass `dim=0` to `torch.mean`. In your own words, what
   does `dim=0` mean? Why does reducing along dimension 0 give you one value per
   column rather than one value per row?

2. `cosine_similarity` returns a value between -1 and 1. Give an example of
   two vectors whose cosine similarity is exactly 0, and explain geometrically
   what that means.

Format your answers like this at the bottom of `linear_algebra.py`:
```python
# ── Reflection ────────────────────────────────────────────────────────────────
# Q1: ...
# Q2: ...
```

---

## Submission

1. Zip your `hw/` folder (include `linear_algebra.py` and nothing else you
   should not modify).
2. Upload the `.zip` to Gradescope under **HW01 — Linear Algebra (Coding)**.
3. Upload your math exercises PDF to Gradescope under **HW01 — Linear Algebra
   (Math)**.

Submit each part separately — two distinct Gradescope submissions.

---

## Time estimate

**4–6 hours** for the coding portion. Budget additional time for the math
exercises.

Part 1 is mostly straightforward once you understand the definitions — the
challenge is translating the math notation into code precisely. Part 2 is
shorter but requires learning how to express the same ideas using PyTorch's API.
`solve_linear_system` is worth the most points; read its docstring carefully.
