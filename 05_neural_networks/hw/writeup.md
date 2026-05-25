# HW04 Writeup — Neural Networks

Run `uv run python analysis.py` to generate `ablation_curves.png` and
`ablation_table.txt`, then answer the questions below. Replace each
`[your answer here]` with your actual response.

---

## Question 1: Reading the Curves

Open `ablation_curves.png` and `ablation_table.txt`. For each of the four
configurations, fill in the final training accuracy and final validation
accuracy from the table:

| Config | Final train loss | Final val acc | Test acc |
|---|---:|---:|---:|
| baseline | | | |
| dropout | | | |
| batchnorm | | | |
| both_schedule | | | |

**a)** Which configuration has the smallest gap between final train loss
and final validation loss? What does this gap measure, in terms from the
lecture (Session B Section "Overfitting and Underfitting")?

[your answer here]

**b)** Which configuration achieves the highest validation accuracy? Is it
the same as the answer to (a)? If they are different, explain in one or
two sentences why "smallest train-val gap" and "highest val accuracy" are
not necessarily the same thing.

[your answer here]

---

## Question 2: The Effect of Dropout

Compare the **baseline** and **dropout** configurations using
`ablation_curves.png`.

**a)** Look at the train loss curves for the two configurations. After 10
epochs, which configuration has the lower training loss? Why does dropout
have this effect? Use the lecture's intuition about dropout's role
(Session B Section "Dropout").

[your answer here]

**b)** Now look at the validation loss curves. Does dropout reduce the
final validation loss, increase it, or leave it roughly the same? Connect
this to the train/val gap idea from Question 1.

[your answer here]

---

## Question 3: The Effect of Batch Normalization

Compare the **baseline** and **batchnorm** configurations.

**a)** Look at the early epochs (say, epochs 1-5) of the training loss
curves. Does batch norm let the model converge faster — that is, reach a
given training loss in fewer epochs? Cite specific numbers from your
curves.

[your answer here]

**b)** The lecture (Session B Section "Batch Normalization") describes two
intuitions for why batch norm helps. Pick one of them and explain why it
predicts the speed-up you observed (or didn't observe) in part (a).

[your answer here]

---

## Question 4: Train Mode vs Eval Mode

This question is about a category of bug the lecture flagged as the
single most common source of silent failures in PyTorch.

**a)** In your `Dropout.forward`, you used `self.training` to decide whether
to apply the dropout mask. What would happen during **validation** if you
forgot to use `self.training` and always applied the mask? Would the
model's accuracy on the val set be higher, lower, or unaffected? Why?

[your answer here]

**b)** Now consider your `MLP` with batch normalization (the
`batchnorm` or `both_schedule` config). What happens if you forget to call
`model.eval()` before running `validate()`? Connect to the lecture's
discussion of batch norm in eval mode versus training mode.

[your answer here]

---

## Question 5: Design Reflection

Suppose you trained the `both_schedule` configuration and still saw clear
overfitting (train loss near zero, val loss climbing).

Name two things you would try **next**, ranked in priority order. For each,
give a one-sentence justification grounded in the lecture's diagnostic
guide (Session B Section "Reading Training Curves").

**Priority 1:**

[your answer here]

**Priority 2:**

[your answer here]
