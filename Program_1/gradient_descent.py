import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

learning_rate = 0.01
epochs = 50

def predict(x, w):
    return w * x

def calculate_loss(x, y, w):
    return np.mean((predict(x, w) - y) ** 2)

# Batch Gradient Descent
def batch_gradient_descent():
    w = 0.0
    losses = []

    for epoch in range(epochs):
        prediction = predict(X, w)
        gradient = np.mean(2 * X * (prediction - y))
        w -= learning_rate * gradient
        losses.append(calculate_loss(X, y, w))

    return losses

# Stochastic Gradient Descent
def stochastic_gradient_descent():
    w = 0.0
    losses = []

    for epoch in range(epochs):
        for i in range(len(X)):
            prediction = predict(X[i], w)
            gradient = 2 * X[i] * (prediction - y[i])
            w -= learning_rate * gradient

        losses.append(calculate_loss(X, y, w))

    return losses

# Mini-Batch Gradient Descent
def mini_batch_gradient_descent():
    w = 0.0
    batch_size = 2
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(len(X))

        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]

            xb = X[batch_indices]
            yb = y[batch_indices]

            prediction = predict(xb, w)
            gradient = np.mean(2 * xb * (prediction - yb))

            w -= learning_rate * gradient

        losses.append(calculate_loss(X, y, w))

    return losses

# Execute algorithms
batch_loss = batch_gradient_descent()
stochastic_loss = stochastic_gradient_descent()
mini_batch_loss = mini_batch_gradient_descent()

# Display results
print("Batch Gradient Descent Loss:", batch_loss[-1])
print("Stochastic Gradient Descent Loss:", stochastic_loss[-1])
print("Mini-Batch Gradient Descent Loss:", mini_batch_loss[-1])

# Plot graph
plt.plot(batch_loss, label="Batch GD")
plt.plot(stochastic_loss, label="Stochastic GD")
plt.plot(mini_batch_loss, label="Mini-Batch GD")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Comparison of Gradient Descent Methods")
plt.legend()
plt.grid()
plt.savefig("../../Results/graphs/gradient_descent.png")
plt.show()