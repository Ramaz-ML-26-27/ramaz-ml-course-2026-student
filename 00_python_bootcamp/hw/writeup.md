# HW00 Writeup — Song Analysis

Run `uv run python analysis.py` to generate the results, then answer the five questions
below. Replace each `[your answer here]` with your response.

- Questions 1–3 are factual. One or two sentences is enough.
- Question 4 asks you to reflect on something that surprised you.
- Question 5 asks you to connect what you implemented in Part 1 to what pandas is doing.

---

## Question 1 (2 pts)

**Which genre averaged the most weeks on the Billboard chart?**

[your answer here]

---

## Question 2 (2 pts)

**Who was the most-streamed artist in the dataset (by total streams across all their songs)?**

[your answer here]

---

## Question 3 (2 pts)

**Which year had the most top-10 hits (songs that peaked at position 10 or better)?**

[your answer here]

---

## Question 4 (4 pts)

**What surprised you about the data?**

Pick one finding from your analysis that was unexpected — something that contradicts what
you assumed going in, or that is more interesting than you expected. Explain:
- What you expected to see, and why.
- What the data actually showed.
- What might explain the difference.

[your answer here]

---

## Question 5 (5 pts)

In Part 1 you implemented `group_by` from scratch using only Python dicts and lists.
Pandas' `groupby()` does something conceptually similar, but at a much larger scale.

Answer both parts:

**a)** What data structure do you think `groupby` uses internally to organize the rows?
Walk through how you think it works step by step, drawing on your own `group_by`
implementation.

**b)** When you call `df.groupby("genre")["weeks_on_chart"].mean()`, what is pandas
doing? Describe each step — how the data gets grouped, and how the mean is computed —
in your own words.

[your answer here]
