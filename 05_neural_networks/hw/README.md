# HW04 — Neural Networks

In this assignment you implement an MLP, a custom inverted-dropout module, and
the full PyTorch training loop. Then you run a four-configuration ablation
study on FashionMNIST and analyze the results.

This assignment is structured CS336-style: tests are the contract, and you
write the implementation to match the spec. The spec lives in `spec.pdf`.

---

## How to work on this assignment

1. **Read `spec.pdf`** end-to-end. It describes every class, function, and
   helper you must implement, with full mathematical detail.
2. **Implement `network.py`** (Dropout + MLP) and get the
   `@pytest.mark.unit` tests green.
3. **Implement `training.py`** (`train_one_epoch`, `validate`, `train`) and
   get the `@pytest.mark.integration` tests green.
4. **Implement `analysis.py`** helpers (`load_fashion_mnist`, `build_model`)
   and get the `@pytest.mark.analysis` tests green.
5. **Run the ablation:** `uv run python analysis.py`. This is teacher-provided
   code that trains all four configurations and saves
   `ablation_curves.png` and `ablation_table.txt`. You do not write this code,
   you just run it once you have implementations in place.
6. **Answer the questions in `writeup.md`** using the plots and table.

```
spec.pdf  →   network.py  →  training.py  →  analysis.py  →  python analysis.py  →   writeup.md
  read         implement       implement       implement         (auto-runs ablation)        answer
```

---

## File inventory

You edit these files:
- `network.py` — `Dropout` and `MLP` classes
- `training.py` — `train_one_epoch`, `validate`, `train` functions
- `analysis.py` — `load_fashion_mnist`, `build_model` (top half only;
  the `__main__` block below the separator is teacher-provided and runs the
  ablation when you invoke `uv run python analysis.py`)
- `writeup.md` — your answers

You do **not** edit these files:
- `spec.pdf` — the spec your code must match (source: `spec.tex`)
- `test_network.py` — the test suite
- `conftest.py` — pytest configuration (seeds, forbidden-import checks)
- `score.py` — local score reporter
- `pyproject.toml`, `uv.lock` — dependency definitions
- the `if __name__ == "__main__":` block in `analysis.py` — runs the ablation

---

## Setup

```bash
uv sync
```

The first time you run `analysis.py` or the analysis tests, FashionMNIST
(~30 MB) will be downloaded to `./data/`. Subsequent runs use the cache.

---

## Running tests

```bash
uv run pytest                      # all tests
uv run pytest -m unit              # unit tests only       (35 pts)
uv run pytest -m integration       # integration tests     (30 pts)
uv run pytest -m analysis          # analysis tests        (35 pts)
```

The analysis tests touch FashionMNIST and are slower than unit/integration
tests; if you want a quick local check, run `unit` and `integration` first.

---

## Checking your score

```bash
uv run python score.py             # full report
uv run python score.py unit        # one section only
```

The `writeup.md` answers are graded separately by the teacher.

---

## Scoring breakdown — 100 pts (autograded)

| Section | Pts | What it tests |
|---|---:|---|
| **Unit** | 35 | `Dropout` (training/eval modes, scaling) and `MLP` (architecture, forward shapes) |
| **Integration** | 30 | `train_one_epoch`, `validate`, full `train()` orchestration with early stopping |
| **Analysis** | 35 | `load_fashion_mnist` (correct splits, shapes) and `build_model` (all four configs) |

Within a section, each test class is **all-or-nothing** — every test in the
class must pass to earn the points for that class.

---

## Restrictions

- **No `torch.nn.Dropout`** anywhere in `network.py`. The whole point of §2
  of the spec is implementing dropout yourself. The conftest detects this
  and fails the run.
- **No `sklearn.neural_network`.** Build the MLP from `torch.nn` primitives.

You may (and should) use:
- `torch.autograd` / `loss.backward()` / `requires_grad` — these are fine now
  and are how `training.py` works.
- `torch.nn.Linear`, `torch.nn.ReLU`, `torch.nn.BatchNorm1d`, `torch.nn.Flatten`,
  `torch.nn.Sequential` — all standard.
- `torch.optim.SGD`, `torch.optim.lr_scheduler.StepLR` — used by
  `build_model` and the training loop.

---

## Generating ablation results (for writeup)

After all autograded tests pass:

```bash
uv run python analysis.py
```

This trains all four configurations (baseline / dropout / batchnorm /
both_schedule) on FashionMNIST and produces:

- `ablation_curves.png` — train and validation loss curves for all four configs
- `ablation_table.txt` — final accuracies and best-epoch summary

Look at these results before writing your answers in `writeup.md`.

---

## Submission

```bash
zip hw04.zip network.py training.py analysis.py writeup.md
```

Upload to **two separate Gradescope submissions**:
1. **HW04 — Code** → upload `hw04.zip`
2. **HW04 — Writeup** → upload `writeup.md` directly

---

## Saving your work

Commit and push regularly so your work isn't lost if your Codespace resets.
See the HW00 README for step-by-step instructions.
