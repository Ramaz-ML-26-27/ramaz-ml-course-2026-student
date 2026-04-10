# HW00 — Python Bootcamp

**Estimated time:** 4–6 hours  
**Prerequisites:** Basic Python familiarity (variables, loops, conditionals, functions)

This assignment reviews the Python patterns we'll rely on throughout the course — lists,
dicts, sets, higher-order functions, and classes — and gives you a first taste of data
analysis with pandas.

---

## Setup

Open this folder in a GitHub Codespace (recommended) or your local editor.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the tests to see where you stand:**
   ```bash
   uv run pytest
   ```

3. **Check your score:**
   ```bash
   uv run python score.py
   ```

4. **Score a specific part only:**
   ```bash
   uv run python score.py lists      # Part 1
   uv run python score.py dicts      # Part 2
   uv run python score.py sets       # Part 3
   uv run python score.py hof        # Part 4
   uv run python score.py classes    # Part 5
   uv run python score.py analysis   # Part 6
   ```

---

## Parts

### Part 1: Python Basics (`python_basics.py`) — 53 pts

Implement 20 functions across five sections:

| Section | Topic | Points |
|---------|-------|--------|
| 1.1 | Lists | 14 |
| 1.2 | Dicts | 13 |
| 1.3 | Sets | 5 |
| 1.4 | Higher-order functions | 11 |
| 1.5 | Classes | 10 |

Read each function's docstring — it tells you exactly what to implement and shows
examples. The examples in the docstring match some of the test cases.

### Part 2: Song Analysis (`analysis.py` + `writeup.md`) — 17 pts (code) + 15 pts (writeup)

Implement the five pandas functions in `analysis.py`. Then run the analysis script to
print results:

```bash
uv run python analysis.py
```

Use the printed output to answer the five questions in `writeup.md`.

---

## Files

| File | What to do |
|------|------------|
| `python_basics.py` | Implement all functions (look for `pass` — replace with your code) |
| `analysis.py` | Implement all five pandas functions |
| `writeup.md` | Answer the five analysis questions in full sentences |

**Do not modify:** `test_python_basics.py`, `test_analysis.py`, `conftest.py`, `pyproject.toml`

---

## Tips

- Work through the sections in order — later problems sometimes build on earlier ones.
- After each function, run `uv run pytest -k TestFunctionName` to test just that function.
- The test failure messages are designed to tell you exactly what went wrong — read them.
- If you're stuck on `analysis.py`, check the pandas docs for `groupby`, `nlargest`, and `idxmax`.

---

## Saving your work

Your Codespace saves automatically while it's open, but it will be deleted if inactive for
30 days. **Commit and push your work to GitHub to keep it permanently.**

You can do this entirely from the VS Code sidebar — no terminal needed:

1. Click the **Source Control** icon in the left sidebar (it looks like a branching tree)
2. Click **+** next to each changed file to stage it
3. Type a short commit message (e.g. `"complete part 1"`)
4. Click **Commit**, then **Sync Changes**

Your work is now saved to your GitHub fork. Get into the habit of doing this whenever
you finish a work session.

---

## Submitting

1. Commit and push your work (see above)
2. Download your `hw/` folder: in the VS Code file explorer, right-click `hw/` → **Download**
3. Submit the downloaded `.zip` on Gradescope
