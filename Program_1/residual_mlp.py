import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.makedirs("../../Results/graphs", exist_ok=True)

# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train[:10000].astype("float32") / 255.0
y_train = y_train[:10000]

X_test = X_test[:2000].astype("float32") / 255.0
y_test = y_test[:2000]

# Plain MLP
def plain_mlp():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model

# Residual MLP
def residual_mlp():

    inputs = tf.keras.Input(shape=(28, 28))

    x = tf.keras.layers.Flatten()(inputs)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    shortcut = x

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128)(x)

    x = tf.keras.layers.Add()([x, shortcut])

    x = tf.keras.layers.Activation("relu")(x)

    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

# Compile and train Plain MLP
model1 = plain_mlp()

model1.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training Plain MLP")

model1.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss1, acc1 = model1.evaluate(X_test, y_test)

# Compile and train Residual MLP
model2 = residual_mlp()

model2.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training Residual MLP")

model2.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss2, acc2 = model2.evaluate(X_test, y_test)

# Display results
print("\nModel Comparison")

print("Plain MLP Accuracy:", acc1 * 100, "%")
print("Residual MLP Accuracy:", acc2 * 100, "%")

# Plot comparison
plt.bar(
    ["Plain MLP", "Residual MLP"],
    [acc1 * 100, acc2 * 100]
)

plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.title("Plain MLP vs Residual MLP")

plt.savefig("../../Results/graphs/residual_comparison.png")
plt.show()