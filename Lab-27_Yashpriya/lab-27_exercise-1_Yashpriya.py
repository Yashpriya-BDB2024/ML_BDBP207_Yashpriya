### Q.1: Can you tell whether the matrix, A = [9  -15;  -15  21] is positive definite?

import numpy as np

"""
A matrix is positive definite if it always gives a positive no. when you sandwich it b/w a non-zero vector & its transpose.
There are 3 conditions to be satisfied for a matrix to be a positive definite:-
1. Matrix must be symmetric (i.e., A = A-transpose).
2. All eigen values of A must be positive.
3. All leading principal minors (determinants of top-left sub-matrices) must be positive.
"""

def is_symmetric(A):
    return np.array_equal(A, A.T)   # Condition-1

def has_positive_eigenvalues(A):   # Condition-2
    eigenvalues = np.linalg.eigvals(A)   # Computes the eigen values of the matrix A.
    return np.all(eigenvalues > 0)  # returns True only if all eigen values are > 0.

def has_positive_leading_minors(A):  # Condition-3
    n = A.shape(0)   # returns the no. of rows (assumes square matrix).
    for k in range(1, n+1):
        minor = A[:k, :k]   # takes the top-left k×k sub-matrix (:k - means rows/columns from index 0 to k-1).
        det = np.linalg.det(minor)   # computes the determinant of the sub-matrix.
        if det <= 0:   # it's not positive definite
            return False
    return True

def main():
    A = np.array([[9, -15], [-15, 21]])
    if is_symmetric(A) and has_positive_eigenvalues(A) and has_positive_leading_minors(A):
        print(f"Matrix {A} is positive definite.")
    else:
        print(f"Matrix {A} is not positive definite.")

if __name__ == "__main__":
    main()
