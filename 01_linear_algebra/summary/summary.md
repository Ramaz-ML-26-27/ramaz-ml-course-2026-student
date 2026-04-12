# Module 1: Linear Algebra — Reference Summary

> This is a distillation, not a replacement for the lecture. Use it as a quick reference while doing the assignment.

---

## Why this matters

- Every data point is a **vector**; every dataset is a **matrix**.
- Predictions in linear regression, forward passes in neural networks, similarity scores in classifiers — all of these are matrix operations.
- Linear algebra is the shared language of data science, statistics, and all of ML.

---

## Vectors

A **vector** $v \in \mathbb{R}^n$ is an ordered list of $n$ real numbers.
$v \in \mathbb{R}^n$ means "$v$ has $n$ entries, each a real number."

$$
v = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

| Operation | Formula | Intuition |
|---|---|---|
| Addition | $(u+v)_i = u_i + v_i$ | Entry-by-entry; parallelogram rule geometrically |
| Scalar multiply | $(cv)_i = c \cdot v_i$ | Stretches (or flips) the vector |
| Dot product | $u \cdot v = \sum_{i=1}^n u_i v_i$ | Multiply entry-by-entry, then **sum** — result is a scalar |

**Dot product is NOT element-wise multiply** — element-wise would give a vector; the dot product gives a single number.

**Worked example:** $u = [1, 2, 3]$, $v = [4, 5, 6]$

$$u \cdot v = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$$

**Geometric meaning:** $u \cdot v = \|u\| \|v\| \cos\theta$, where $\theta$ is the angle between them. Large dot product means the vectors point in similar directions. Zero dot product means they are perpendicular.

---

## Norm and Distance

The **Euclidean norm** (length) of $v \in \mathbb{R}^n$:

$$\|v\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^n v_i^2}$$

The **Euclidean distance** between two vectors:

$$d(u, v) = \|u - v\| = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$

**Normalization** — make $v$ into a unit vector (length = 1) pointing the same direction:

$$\hat{v} = \frac{v}{\|v\|}$$

**Example:** Normalize $v = [3, 4]$.
1. $\|v\| = \sqrt{9 + 16} = \sqrt{25} = 5$
2. $\hat{v} = \frac{1}{5}[3, 4] = [0.6, 0.8]$
3. Check: $\|[0.6, 0.8]\| = \sqrt{0.36 + 0.64} = 1$ (correct)

**Why normalize?** Features measured on different scales (e.g., height in cm vs. income in thousands) can unfairly dominate computations. Normalization puts features on equal footing, which stabilizes learning.

---

## Matrices

A **matrix** $A \in \mathbb{R}^{m \times n}$ is a rectangular array with $m$ rows and $n$ columns. Entry in row $i$, column $j$ is written $A_{ij}$.

### Transpose

$(A^\top)_{ij} = A_{ji}$ — flip rows and columns.

$$\begin{bmatrix} 0 & 4 \\ 7 & 0 \\ 3 & 1 \end{bmatrix}^\top = \begin{bmatrix} 0 & 7 & 3 \\ 4 & 0 & 1 \end{bmatrix}$$

### Matrix-vector multiply

$A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$ — result $Ax \in \mathbb{R}^m$

**Rule:** Entry $i$ of the output = **dot product** of row $i$ of $A$ with $x$.

$$(Ax)_i = \sum_{j=1}^n A_{ij} x_j$$

**Shape rule:** inner dimensions must match. $(m \times n) \cdot (n) = (m)$.

### Matrix-matrix multiply

$A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$ — result $AB \in \mathbb{R}^{m \times p}$

**Rule:** Entry $(i,j)$ of $AB$ = dot product of **row $i$ of $A$** with **column $j$ of $B$**.

$$(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$

**Shape rule:** $(m \times n) \cdot (n \times p) = (m \times p)$. Inner dimensions must match.

**$AB \neq BA$ in general.** Matrix multiplication is **not commutative**. Order matters.

**Deepest interpretation:** $AB$ means "apply $B$ first, then $A$." Matrix multiplication = function composition.

---

## Rotation Matrix (correct formula)

To rotate a 2D vector counterclockwise by angle $\theta$:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Check: $R(0) = I$ (identity). $R(90\degree) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.

---

## Solving $Ax = b$

Given matrix $A \in \mathbb{R}^{n \times n}$ and vector $b \in \mathbb{R}^n$, find $x \in \mathbb{R}^n$ such that $Ax = b$.

- Conceptually: if $A$ is invertible, then $x = A^{-1}b$.
- **In practice: never compute $A^{-1}$ directly.** Use a solver instead.
- **Solving $Ax = b$ is NOT "dividing by $A$."** Division by matrices is not defined; we multiply by the inverse on the left (when it exists).

In PyTorch: `torch.linalg.solve(A, b)` returns $x$.

---

## PyTorch Cheat Sheet

```python
import torch

# Create tensors
v = torch.tensor([1.0, 2.0, 3.0])
u = torch.tensor([4.0, 5.0, 6.0])

# Vector operations
torch.dot(u, v)                # dot product -> scalar tensor; call .item() for float
torch.linalg.norm(v)           # Euclidean norm -> scalar tensor; call .item() for float

# Matrices
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
A @ B                          # matrix multiply (also works for matrix @ vector)
A.mT                           # transpose (works for 2-D and batched tensors)

# Reductions along a dimension
# dim=0: collapse rows -> one value per COLUMN (shape goes from (m,n) to (n,))
# dim=1: collapse columns -> one value per ROW   (shape goes from (m,n) to (m,))
torch.mean(A, dim=0)           # mean along rows -> shape (n,)

# keepdim=True: retains the reduced axis as size 1, which enables broadcasting
# (useful when the result needs to be broadcast back against the original shape)

# Solving Ax = b
b = torch.tensor([7.0, 1.0])
x = torch.linalg.solve(A, b)  # finds x such that A @ x == b
```

---

## Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring their lengths:

$$\text{cos\_sim}(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}$$

- Range: $[-1, 1]$
- $= 1$: same direction (most similar)
- $= 0$: perpendicular (unrelated)
- $= -1$: opposite direction (most dissimilar)

Used everywhere in ML: word embeddings, recommendation systems, image retrieval.

---

## Gram Matrix

For a matrix $A \in \mathbb{R}^{m \times n}$, the **Gram matrix** is:

$$G = A^\top A \in \mathbb{R}^{n \times n}$$

- Entry $G_{ij}$ = dot product of column $i$ of $A$ with column $j$ of $A$
- Always square ($n \times n$) and symmetric ($G = G^\top$)
- Appears in linear regression (normal equations: $A^\top A \, x = A^\top b$)

---

## What You Need for the Assignment

**Part 1 — From Scratch**

| Function | Concepts Needed |
|---|---|
| Vector operations | Addition, scalar multiply, dot product |
| Norm and normalization | $\|v\| = \sqrt{\sum v_i^2}$, $\hat{v} = v / \|v\|$ |
| Matrix-vector multiply | Row-dot-product rule; shape $(m \times n)(n) = (m)$ |
| Matrix-matrix multiply | Entry $(i,j)$ = row $i$ dot column $j$; shape rule |
| Matrix transpose | $T_{ij} = A_{ji}$; swap rows and columns |

**Part 2 — PyTorch**

| Function | What it computes |
|---|---|
| `tensor_dot_product` | dot product of two 1-D tensors → Python float |
| `tensor_magnitude` | Euclidean norm of a 1-D tensor → Python float |
| `tensor_normalize` | unit vector in the direction of v |
| `tensor_matmul` | matrix product of two 2-D tensors |
| `tensor_transpose` | transpose of a 2-D or batched tensor |
| `column_means` | mean of each column → 1-D tensor of length n |
| `row_normalize` | normalize each row to unit length |
| `cosine_similarity` | cosine similarity of two 1-D tensors → Python float |
| `gram_matrix` | Gram matrix G = A^T A |
| `solve_linear_system` | solution x to the system Ax = b |

---

*Reference: Lecture 1 notes. For derivations and worked examples, see the full lecture.*
