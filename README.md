

# 📊 Logistic Regression from Scratch (Python + NumPy)

This project implements **Logistic Regression**, a fundamental binary classification algorithm, entirely from scratch using Python and `NumPy`. It demonstrates training with gradient descent, prediction with the sigmoid activation, and visualization of the decision boundary.

---

## 🔍 Project Overview

Logistic Regression models the probability that an input belongs to a class using the logistic sigmoid function. It is widely used for binary classification problems.

- Programming Language: **Python**
- Libraries: `NumPy`, `Matplotlib`, `Pandas`
- Features:
  - Synthetic dataset generation
  - Model training using gradient descent
  - Probability-based predictions with sigmoid activation
  - Decision boundary visualization

---

## 🔢 Mathematical Formulation

The model computes the linear combination:

\[
z = \mathbf{w} \cdot \mathbf{x} + b
\]

The predicted probability is:

\[
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

Weights and bias are updated via gradient descent minimizing the log-loss:

\[
\mathbf{w} = \mathbf{w} - \alpha (\hat{y} - y) \mathbf{x}
\]

\[
b = b - \alpha (\hat{y} - y)
\]

Where:
- \( \alpha \) is the learning rate,
- \( y \) is the true label (0 or 1),
- \( \hat{y} \) is the predicted probability.

---

## 🛠️ How to Run

1. Install dependencies:
```bash
pip install numpy matplotlib pandas
```

2. Run the script:
```bash
python logistic_regression.py
```

3. (Optional) Save the decision boundary plot by adding:
```python
plt.savefig("logistic_decision_boundary.png")
```
before `plt.show()` in the plotting function.

---

## 📈 Performance

On a synthetic linearly separable dataset, the model typically achieves high accuracy:

```
Accuracy: 99.85%
```

---

## 📷 Output Example

Below is an example plot showing the decision boundary learned by the logistic regression model:

![Logistic Regression Decision Boundary](logistic_decision_boundary.png)

---

## 📂 File Structure

```
.
├── logistic_regression.py    # Main implementation script
├── logistic_decision_boundary.png  # Saved decision boundary plot (optional)
└── README.md                 # This documentation file
```

---

## 🧑‍💻 Author

**Yassine Ouali**  
Contact me for questions, improvements, or collaboration.

---

## 📜 License

This project is released under the MIT License. Feel free to use and adapt it!

```

---

### Pro tip:

To generate and save the plot image, add `plt.savefig("logistic_decision_boundary.png")` right before `plt.show()` in your `plot_decision_boundary` function.

If you'd like, I can also help you write a combined README for both Perceptron and Logistic Regression!
