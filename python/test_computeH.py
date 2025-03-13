import numpy as np
from planarH import computeH

x1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
x2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])

H2to1 = computeH(x1, x2)
print("Computed Homography (H2to1):\n", H2to1)

for i in range(len(x2)):
    x2_hom = np.array([x2[i][0], x2[i][1], 1])
    x1_mapped = H2to1 @ x2_hom
    x1_mapped /= x1_mapped[2]
    print(f"x2[{i}] = {x2[i]} maps to {x1_mapped[:2]}, expected {x1[i]}")