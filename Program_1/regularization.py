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

# Model creation
def create_model(technique):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(28, 28)))

    if technique == "Augmentation":
        model.add(tf.keras.layers.RandomRotation(0.05))
        model.add(tf.keras.layers.RandomTranslation(0.05, 0.05))

    model.add(tf.keras.layers.Flatten())

    if technique == "L2":
        model.add(tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
    else:
        model.add(tf.keras.layers.Dense(128, activation="relu"))

    if technique == "Dropout":
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

techniques = ["None", "Dropout", "L2", "Augmentation"]

results = {}

for technique in techniques:

    print("\nTraining:", technique)

    model = create_model(technique)

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    results[technique] = accuracy * 100

    print("Test Accuracy:", accuracy * 100)

# Display results
print("\nRegularization Comparison")

for technique, accuracy in results.items():
    print(technique, ":", round(accuracy, 2), "%")

# Plot results
plt.bar(results.keys(), results.values())

plt.xlabel("Regularization Techniques")
plt.ylabel("Test Accuracy (%)")
plt.title("Regularization Comparison")

plt.savefig("../../Results/graphs/regularization.png")
plt.show()