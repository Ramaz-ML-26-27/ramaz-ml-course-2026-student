# Ramaz ML Course 2026 — Student Repository

Welcome! This repo holds every assignment for the course. Each module is in its
own folder (`00_python_bootcamp/`, `01_linear_algebra/`, …). Inside each module,
`hw/` is the assignment and (where applicable) `exercises/` is the written
problem set.

You'll work in **GitHub Codespaces** (recommended — browser-based VS Code, no
local installation needed) or on your **local machine** (see below), and submit
on **Gradescope**.

---

## First-time setup

You only do this once, at the start of the year.

### 1. Fork this repo

Click the **Fork** button in the top-right of this repo's GitHub page. This
creates your own copy under your GitHub account. All your work lives on **your
fork** — the upstream repo is read-only to you.

### 2. Launch a Codespace from your fork

On your fork's page:

1. Click the green **`< > Code`** button
2. Switch to the **Codespaces** tab
3. Click **Create codespace on main**

Wait 1–2 minutes for the environment to build. The devcontainer automatically
installs Python 3.11, `uv`, and all dependencies for every released assignment.
When it finishes you'll be in a VS Code editor in your browser.

### 3. Open the terminal

In VS Code: **View → Terminal** (or press `` Ctrl+` `` / `` Cmd+` ``). All the
commands in each assignment's README go in this terminal.

### 4. Verify the setup

Navigate into the first assignment folder and run the tests:

```bash
cd 00_python_bootcamp/hw
uv run pytest
```

If you see a list of failing tests (with names like `test_flatten`,
`test_sliding_window`, …), the environment works. Failing tests are expected —
you haven't implemented anything yet.

---

## Alternative: working on your local machine

If you'd rather use your own computer, the workflow is the same but with a
one-time local setup.

1. **Fork** the repo (as in step 1 above).
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/ramaz-ml-course-2026-student.git
   cd ramaz-ml-course-2026-student
   ```
3. **Install Python 3.11+** if you don't already have it:
   [python.org/downloads](https://www.python.org/downloads/).
4. **Install `uv`** (the Python package manager the course uses):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   On Windows, see
   [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/).
5. **Open the folder in your editor** (VS Code, PyCharm, whatever you like) and
   work from its terminal. The per-assignment `uv run pytest` and `uv run
   python score.py` commands work identically to Codespaces.

Everything else in this README (Sync fork, the two-submission Gradescope flow,
Source Control for saving work) applies the same way locally.

---

## Getting new assignments (during the year)

The teacher releases new assignments by pushing them to the upstream repo. To
pull them into your fork, click **Sync fork** at the top of your fork's page on
GitHub. That's it — one click, no terminal command.

Then bring the changes into your working copy (Codespace or local) and install
any new dependencies:

```bash
git pull origin main
bash setup.sh
```

In a Codespace, just run `bash setup.sh` — `git pull` is done for you by the
sync.

---

## Submitting on Gradescope

You're already enrolled in the Gradescope course. Log in at
[gradescope.com](https://www.gradescope.com) — the course is on your
dashboard, with every assignment listed.

**Most assignments have two separate Gradescope submissions:**

| Submission | What you upload |
|---|---|
| **HWXX — Code** | A `.zip` of your `hw/` folder |
| **HWXX — Writeup** | Your completed `writeup.md` directly |

To create the `.zip` of your `hw/` folder from inside Codespaces:

1. In the file explorer sidebar, **right-click the `hw/` folder**
2. Choose **Download…**
3. Codespaces packages the folder as a `.zip` and saves it to your computer

Upload that `.zip` to the **Code** submission on Gradescope. Upload your
`writeup.md` directly to the **Writeup** submission.

After you submit the code, Gradescope runs the autograder and shows your score
within a couple of minutes. You can resubmit as many times as you like before
the deadline — only the last submission counts.

---

## Saving your work

**Important:** your Codespace will be deleted if inactive for 30 days. Commit
and push your work to GitHub regularly to keep it permanently.

You can do this entirely from the VS Code sidebar — no terminal needed:

1. Click the **Source Control** icon in the left sidebar (it looks like a
   branching tree)
2. Click **+** next to each changed file to stage it
3. Type a short commit message (e.g. `"complete part 1"`)
4. Click **Commit**, then **Sync Changes**

Your work is now saved to your GitHub fork. Get into the habit of doing this
whenever you finish a work session.

---

## Where to start

Open `00_python_bootcamp/hw/README.md`. Each assignment's `README.md` has the
operational details (setup, run, score, submit) and points you at the
assignment content.

---

## Troubleshooting

- **"My Codespace won't open / is stuck building"** — wait 2–3 minutes the first
  time. If it still fails, try **Codespaces → Delete** on the broken Codespace,
  then create a new one.
- **"`uv` is not found"** — close the terminal and open a new one (the install
  added `uv` to your PATH; new terminals pick it up).
- **"Tests can't find a module / `ModuleNotFoundError`"** — run `uv sync` in the
  affected `hw/` folder.
- **"I synced the fork but I don't see the new assignment in Codespaces"** —
  run `bash setup.sh` from the repo root in your Codespace terminal.

For anything else, ask the teacher in class or via Schoology.
