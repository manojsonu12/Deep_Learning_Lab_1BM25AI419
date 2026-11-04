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

# Model
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

# Constant learning rate
model1 = create_model()

model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model1.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Scheduled learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

model2 = create_model()

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model2.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate models
loss1, acc1 = model1.evaluate(X_test, y_test, verbose=0)
loss2, acc2 = model2.evaluate(X_test, y_test, verbose=0)

print("Constant LR Accuracy:", acc1 * 100)
print("Scheduled LR Accuracy:", acc2 * 100)

# Plot comparison
plt.plot(history1.history["val_accuracy"], label="Constant LR")
plt.plot(history2.history["val_accuracy"], label="Scheduled LR")

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid()

plt.savefig("../../Results/graphs/learning_rate.png")
plt.show()