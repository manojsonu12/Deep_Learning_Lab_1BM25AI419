import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.makedirs("../../Results/graphs", exist_ok=True)

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train[:10000].astype("float32") / 255.0
y_train = y_train[:10000]

X_test = X_test[:2000].astype("float32") / 255.0
y_test = y_test[:2000]

# Model function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model

# Optimizers
optimizers = {
    "SGD Momentum": tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9
    ),
    "RMSprop": tf.keras.optimizers.RMSprop(
        learning_rate=0.001
    ),
    "Adam": tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    "Adagrad": tf.keras.optimizers.Adagrad(
        learning_rate=0.01
    )
}

results = {}

# Train each optimizer
for name, optimizer in optimizers.items():

    print("\nTraining using", name)

    model = create_model()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    results[name] = accuracy * 100

    print("Test Accuracy:", accuracy * 100)

# Display comparison
print("\nOptimizer Comparison")

for name, accuracy in results.items():
    print(name, ":", round(accuracy, 2), "%")

# Plot graph
plt.bar(results.keys(), results.values())

plt.xlabel("Optimizers")
plt.ylabel("Test Accuracy (%)")
plt.title("Optimizer Comparison")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("../../Results/graphs/optimizer_comparison.png")
plt.show()