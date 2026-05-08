# HW03 Writeup — Linear and Logistic Regression

Run `uv run python analysis.py` to generate your results, then answer the
questions below. Replace each `[your answer here]` with your actual response.

---

## Question 1 (2 pts): Learning Rate Effects

**Run `analysis.py`. It trains your linear regression model with three learning
rates: `lr=0.01`, `lr=0.5`, and `lr=1e-5`. Look at the `loss_curve.png` plot
and the printed loss values.**

**a)** Describe what the loss curve looks like for each learning rate.
What happens to the loss over the 200 epochs in each case?

[your answer here]

**b)** Your Calculus lecture (Lesson 2, Section 3.2) described three cases:
"too large", "just right", and "too small" learning rates.
Which of your three runs corresponds to each case?
Use specific numbers from your printed output to support your answer.

[your answer here]

---

## Question 2 (2 pts): The Decision Boundary

**Look at your `decision_boundary.png` and the printed learned parameters
`w = [w1, w2]` and `b`.**

**a)** The decision boundary is the line where $\hat{y} = 0.5$, which means
$\sigma(x_1 w_1 + x_2 w_2 + b) = 0.5$.
Since $\sigma(z) = 0.5$ exactly when $z = 0$, the boundary satisfies:

$$x_1 w_1 + x_2 w_2 + b = 0$$

Using your actual trained values of $w_1$, $w_2$, and $b$, rewrite this as a
line equation in the form $x_2 = m x_1 + c$.
What are $m$ and $c$ in terms of $w_1$, $w_2$, and $b$?

[your answer here]

**b)** Does the decision boundary in your plot correctly separate the two
classes? Are there any misclassified points? What does the test accuracy tell
you about how well the model generalized?

[your answer here]

---

## Question 3 (3 pts): Why the Gradients Look the Same

**In the lecture, you derived two gradient formulas that look nearly identical:**

$$\text{MSE gradient: } \frac{\partial L}{\partial \mathbf{w}} = \frac{2}{n} X^\top (\hat{\mathbf{y}} - \mathbf{y})$$

$$\text{BCE gradient: } \frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} X^\top (\hat{\mathbf{y}} - \mathbf{y})$$

**a)** In `LinearRegression.gradient`, your formula has a factor of 2. Where
does this factor come from? Trace it back to the derivative of the squared
error term $(\hat{y}_i - y_i)^2$ with respect to $w_j$.

[your answer here]

**b)** The BCE gradient has no factor of 2. Look at the BCE gradient derivation
in your lecture notes (Lesson 4, Section 2.5). Explain in one or two sentences
why the factor of 2 does not appear for BCE. What cancels it out?

[your answer here]

---

## Question 4 (3 pts): Design Reflection

**You implemented `LinearRegression` and `LogisticRegression` as classes. The
spec told you what methods to write but not how to organize the internals —
that was your call.**

**a)** What instance attributes (the things on `self`) did you store, and why?
Could any of them have been local variables instead? What changes if so?

[your answer here]

**b)** Inside your `fit` method, you call `self.forward`, `self.loss`, and
`self.gradient`. Why is it useful that these are separate methods rather than
a single big function? Give a concrete example of something that would be
harder if they were merged.

[your answer here]
