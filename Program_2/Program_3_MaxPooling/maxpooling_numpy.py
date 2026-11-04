import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "Results" / "MaxPooling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Max Pooling function
def max_pooling(image, pool_size=2, stride=2):

    image = np.array(image, dtype=float)

    h, w = image.shape

    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    if oh <= 0 or ow <= 0:
        raise ValueError("Pooling window is larger than input")

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):

            region = image[
                i * stride:i * stride + pool_size,
                j * stride:j * stride + pool_size
            ]

            output[i, j] = np.max(region)

    return output


# Input feature map
image = np.array([
    [1, 3, 2, 4, 5, 1],
    [5, 6, 1, 2, 3, 2],
    [7, 2, 8, 4, 1, 9],
    [1, 3, 2, 6, 7, 2],
    [4, 5, 1, 8, 2, 3],
    [2, 1, 9, 3, 5, 4]
])

print("INPUT FEATURE MAP")
print(image)

# 2x2 Max Pooling
output1 = max_pooling(image, 2, 2)

print("\n2x2 Max Pooling:")
print(output1)

# 2x2 pooling with stride 1
output2 = max_pooling(image, 2, 1)

print("\n2x2 Pooling with Stride 1:")
print(output2)

# 3x3 Max Pooling
output3 = max_pooling(image, 3, 2)

print("\n3x3 Max Pooling:")
print(output3)

# Display results
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].imshow(image, cmap="viridis")
axes[0].set_title("Original Feature Map")

axes[1].imshow(output1, cmap="viridis")
axes[1].set_title("2x2 Max Pooling")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "maxpooling_output.png",
    dpi=150
)

plt.show()

print("\nMax Pooling program completed!")