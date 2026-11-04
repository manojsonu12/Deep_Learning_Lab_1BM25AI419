import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Output folder
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "Results" / "CIFAR10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 classes
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

print("LOADING CIFAR-10 DATASET")

# Load dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.flatten()
y_test = y_test.flatten()

print("\nDataset loaded successfully!")

# Display sample images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_images.png")
plt.show()


# Build CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(10, activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN MODEL SUMMARY")
model.summary()


# Train model
print("\nTRAINING CNN MODEL")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


# Evaluate model
print("\nMODEL EVALUATION")

test_loss, test_accuracy = model.evaluate(
    x_test, y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# Accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "accuracy_graph.png")
plt.show()


# Loss graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "loss_graph.png")
plt.show()


# Predictions
predictions = model.predict(x_test[:10], verbose=0)

predicted_classes = np.argmax(predictions, axis=1)

print("\nSAMPLE PREDICTIONS")

for i in range(10):

    print(
        "Actual:", class_names[y_test[i]],
        "| Predicted:", class_names[predicted_classes[i]]
    )


# Display prediction images
plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i])

    plt.title(
        "Predicted: " + class_names[predicted_classes[i]]
        + "\nActual: " + class_names[y_test[i]],
        fontsize=8
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions.png")
plt.show()


# Confusion Matrix
y_predictions = model.predict(x_test, batch_size=128, verbose=1)

y_pred = np.argmax(y_predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CIFAR-10 Confusion Matrix")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()


# Classification Report
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)


# Save trained model
model.save(OUTPUT_DIR / "cifar10_cnn.keras")

print("\nCNN MODEL SAVED SUCCESSFULLY!")
print("All results saved in:", OUTPUT_DIR)

print("\nPROGRAM COMPLETED!")