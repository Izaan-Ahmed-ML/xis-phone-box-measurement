import torch
import torch.nn as nn



#createing fake image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

image = np.zeros((8, 8), dtype=np.uint8)


# plt.imshow(image, cmap=cmap,vmin=0, vmax=255)
# plt.show()

#mask
mask = np.zeros((8, 8), dtype=np.uint8)

mask[2:6, 2:6] = 1
cmap = ListedColormap(["blue","yellow"])
# plt.imshow(mask, cmap=cmap,vmin=0, vmax=255 )
# plt.show()

partial_mask = mask.copy()

partial_mask[0:2, :] = 255
partial_mask[6:8, :] = 255
mask[2:6, 2:6] = 1

# plt.imshow(partial_mask, cmap=cmap,vmin=0, vmax=255)
# plt.show()
print(partial_mask)


# 1 image
# 2 classes
# 8x8 pixels

# input = torch.randn(1, 2, 8, 8, requires_grad=True)
# print(input.shape)

# target = torch.tensor(partial_mask, dtype=torch.long).unsqueeze(0)

# print("input:", input.shape)
# print("target:", target.shape)

# loss = nn.CrossEntropyLoss(ignore_index=255)

# output = loss(input, target)

# output.backward()

# print("loss:", output)


import torch
import torch.nn as nn

# 1 image, 2 classes, 2x2 pixels
input = torch.tensor([
    [
        [2., 1.],   # Class 0 logits
        [4., 1.]
    ],
    [
        [1., 3.],   # Class 1 logits
        [2., 2.]
    ]
])

# Add batch dimension becasue segmentation input requires shape (N, C, H, W)
input = input.unsqueeze(0)

# Target:
# 0 = background
# 1 = object
# 255 = unknown / ignore

target = torch.tensor([
    [0, 1],
    [255, 1]
]).unsqueeze(0)

print("Input shape:", input.shape)
print("Target shape:", target.shape)

# Partial Cross Entropy
loss_function = nn.CrossEntropyLoss()

loss = loss_function(input, target)

print("Partial CE:", loss)