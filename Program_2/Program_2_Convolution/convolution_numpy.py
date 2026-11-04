import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "Results" / "Convolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Convolution function
def convolution(image, kernel, stride=1, padding=0):

    image = np.array(image, dtype=float)
    kernel = np.array(kernel, dtype=float)

    # Apply zero padding
    image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant"
    )

    kh, kw = kernel.shape

    oh = (image.shape[0] - kh) // stride + 1
    ow = (image.shape[1] - kw) // stride + 1

    output = np.zeros((oh, ow))

    # Perform convolution
    for i in range(oh):
        for j in range(ow):

            region = image[
                i * stride:i * stride + kh,
                j * stride:j * stride + kw
            ]

            output[i, j] = np.sum(region * kernel)

    return output


# Input image
image = np.array([
    [1, 2, 0, 3, 1],
    [4, 1, 2, 1, 0],
    [0, 3, 1, 2, 2],
    [2, 2, 0, 1, 3],
    [1, 0, 2, 3, 1]
])

# 3x3 kernel
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

print("INPUT IMAGE")
print(image)

print("\nKERNEL")
print(kernel)

# Different experiments
output1 = convolution(image, kernel, stride=1, padding=0)
output2 = convolution(image, kernel, stride=2, padding=0)
output3 = convolution(image, kernel, stride=1, padding=1)

print("\nStride 1, Padding 0:")
print(output1)

print("\nStride 2, Padding 0:")
print(output2)

print("\nStride 1, Padding 1:")
print(output3)

# Different kernel size
kernel2 = np.ones((2, 2))

output4 = convolution(image, kernel2)

print("\n2x2 Kernel Output:")
print(output4)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

data = [
    (image, "Input Image"),
    (kernel, "3x3 Kernel"),
    (output1, "Convolution"),
    (output3, "With Padding")
]

for ax, (matrix, title) in zip(axes, data):
    ax.imshow(matrix, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "convolution_output.png",
    dpi=150
)

plt.show()

print("\nConvolution program completed!")