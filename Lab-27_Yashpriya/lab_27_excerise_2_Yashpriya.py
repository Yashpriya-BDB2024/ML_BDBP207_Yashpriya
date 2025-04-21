### Q.2: Find the eigenvalues of the given Hessian at the given point [12x^2  -1;  -1   2] at (3, 1).

import numpy as np

"""
About Hessian:
It tells us how the curvature (bending) of a function behaves in multiple dimensions.
For a function f(x,y), in Hessian matrix, the diagonal elements are second partial derivatives w.r.t. each variable, and the off-diagonal elements are mixed partial derivatives.
It tells us how the slope of f(x,y) is changing.
1. If the Hessian is positive definite at a point, the function has a local minimum there.
2. If it's negative definite, it's a local maximum.
3. If it's indefinite, then gradient is zero.
"""

# Method-1 -

def find_eigenvalues_2by2(a,b,c,d):
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace**2 - 4 * determinant
    sqrt_disc = discriminant**0.5
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    return lambda1, lambda2

def main():
    x = 3  # Given point: (3,1)
    a = 12*x**2
    b = -1
    c = -1
    d = 2
    eig1, eig2 = find_eigenvalues_2by2(a,b,c,d)
    print("Eigenvalues of Hessian at (3,1):", eig1, eig2)
    if eig1>0 and eig2>0:
        print("Hessian is positive definite at (3,1).")
    else:
        print("Hessian is not positive definite at (3,1).")

if __name__ == "__main__":
    main()

# Method-2 -

# H = np.array([[12*3**2, -1], [-1, 2]])   # x=3
# eigvals = np.linalg.eigvals(H)
# print("Eigenvalues of Hessian at (3,1):", eigvals)
# if np.all(eigvals > 0):
#     print("Hessian is positive definite at (3,1).")
# else:
#     print("Hessian is negative definite at (3,1).")
