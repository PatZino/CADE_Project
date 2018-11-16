import numpy as np
from numpy import array


A = array([[1, 3, 0], [3, 2, 0], [0, 2, 1], [1, 1, 4], [3, 2, 2], [0, 1, 0], [1, 3, 1], [0, 4, 1], [2, 4, 2],
           [3, 3, 1]])

idx = np.random.randint(10, size=2)
print("idx: ", idx, "\n")

print("A[idx, :] : ", A[idx, :])

print("A.shape: ", A.shape)

print("A[np.random.randint(A.shape[0], size=2), :] : ", A[np.random.randint(A.shape[0], size=2), :])

print("A[np.random.choice(A.shape[0], 2, replace=False), :] : ", A[np.random.choice(A.shape[0], 2, replace=False), :])

