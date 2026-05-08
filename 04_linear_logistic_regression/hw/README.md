# HW03 — Linear and Logistic Regression

In this assignment you implement the two most fundamental supervised learning
models from scratch as **classes**, computing every gradient by hand using the
formulas derived in Lesson 4. No `torch.autograd` here — you are the autograd
engine.

This assignment is structured CS336-style: tests are the contract, and you
write the implementation to match the spec. The spec lives in `spec.pdf`.

---

## How to work on this assignment

1. **Read `spec.pdf`** end-to-end. It describes the architecture, the public API
   you must implement, and the math for every method.
2. **Implement methods unit-test-first.** Open `regression.py`. Get the
   `@pytest.mark.unit` tests green one class at a time:
   `sigmoid` → `LinearRegression` (init → forward → loss → gradient) →
   `LogisticRegression` (same order).
3. **Run `pipeline.py`** as soon as the unit tests pass. It trains both models
   on synthetic data and prints loss + accuracy. If those look right, your
   pipeline works end-to-end.
4. **Get the integration tests green.** These verify that `fit` actually
   converges and `predict` produces sensible outputs.
5. **Complete `analysis.py`.** Implement the four helper functions and run
   `uv run python analysis.py` to generate plots for your writeup.
6. **Answer the questions in `writeup.md`** using the plots and printed numbers.

```
spec.pdf  →   regression.py   →   pipeline.py   →   tests   →   analysis.py   →   writeup.md
  read         implement           sanity check       grade        run             answer
```

---

## File inventory

You edit these files:
- `regression.py` — your implementations
- `analysis.py` — your implementations of the four helper functions
- `writeup.md` — your answers

You do **not** edit these files (they are part of the assignment infrastructure):
- `spec.pdf` — the spec your code must match (source: `spec.tex`)
- `test_regression.py` — the test suite
- `conftest.py` — pytest configuration (seeds, no-autograd enforcement)
- `pipeline.py` — sanity-check demo
- `score.py` — local score reporter
- `pyproject.toml`, `uv.lock` — dependency definitions

---

## Setup

```bash
uv sync
```

---

## Running tests

```bash
uv run pytest                      # all tests, full output
uv run pytest -m unit              # unit tests only       (40 pts)
uv run pytest -m integration       # integration tests     (30 pts)
uv run pytest -m analysis          # analysis tests        (20 pts)
```

---

## Checking your score

```bash
uv run python score.py             # full report
uv run python score.py unit        # one section only
```

`writeup.md` (10 pts) is graded manually and is not part of the autograded score.

---

## Scoring breakdown — 100 pts

| Section | Pts | What it tests |
|---|---:|---|
| **Unit** | 40 | Every method (`forward`, `loss`, `gradient`, `__init__`, `sigmoid`) tested in isolation |
| **Integration** | 30 | `fit` converges; `predict` and `predict_proba` produce correct outputs end-to-end |
| **Analysis** | 20 | The four helper functions in `analysis.py` |
| **Writeup** | 10 | Four short-answer questions in `writeup.md` (graded manually) |

Within a section, each test class is **all-or-nothing** — every test in the
class must pass to earn the points for that class. This rewards getting things
fully right and discourages "good enough" partial implementations.

---

## Restrictions

- **No `torch.autograd`, no `.backward()`, no `requires_grad=True`** anywhere
  in `regression.py`. The conftest fails the run if any of these appear.
- **No `sklearn.linear_model`.** `analysis.py` may use `sklearn.datasets`,
  `train_test_split`, and `StandardScaler` — those help with data loading but
  do not bypass the learning goal.

---

## Generating analysis results (for writeup)

```bash
uv run python analysis.py
```

Produces:
- `loss_curve.png` — training loss curves for three learning rates
- `decision_boundary.png` — logistic regression decision boundary on Iris

Look at these plots before writing your answers in `writeup.md`.

---

## Submission

```bash
zip hw03.zip regression.py analysis.py writeup.md
```

Upload to **two separate Gradescope submissions**:
1. **HW03 — Code** → upload `hw03.zip`
2. **HW03 — Writeup** → upload `writeup.md` directly

---

## Saving your work

Commit and push regularly so your work isn't lost if your Codespace resets.
See the HW00 README for step-by-step instructions.
