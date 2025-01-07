import numpy as np

# Finding the transpose of the given matrix -
original_mat = np.matrix([[1, 2, 3],
                             [4, 5, 6]])
transposed_mat = original_mat.transpose()
print("Original Matrix:")
print(original_mat)
print("Transposed Matrix")
print(transposed_mat)

# Multiplying the transposed matrix with the given matrix -
print("Product of original matrix and transposed matrix: ")
print(transposed_mat @ original_mat)
