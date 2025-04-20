### Q-4: f(x, y) = 4x + 2y - x^2 - 3y^2. Find the gradient. Use that to find critical points, (x, y) that makes gradient 0.
### Use the Eigenvalues of the Hessian at the point to determine whether the critical point is a minimum, maximum or neither.

import sympy as sp
from lab_27_excerise_2_Yashpriya import find_eigenvalues_2by2
from lab_27_exercise_3_Yashpriya import compute_second_derivatives, construct_hessian

x, y = sp.symbols('x y')   # Declares symbolic variables x and y. These are used to define and manipulate algebraic expressions.
f_expr = 4*x + 2*y - x**2 - 3*y**2

fx_expr = sp.diff(f_expr, x)   # Partial derivative of function f_expr w.r.t. x
fy_expr = sp.diff(f_expr, y)   # Partial derivative of function f_expr w.r.t. y

critical_points = sp.solve([fx_expr, fy_expr], (x, y), dict=True)  # Solves the gradient equations ; fx_expr = 0 and fy_expr = 0 ; sp.solve() returns a list of dictionaries.

if critical_points:
    for point in critical_points:
        x_c = point[x]   # Unpacks each critical point found.
        y_c = point[y]
        print(f"\nCritical point: ({x_c}, {y_c})")
        f_xx, f_yy, f_xy, f_yx = compute_second_derivatives(f_expr, x_c, y_c)    # All second-order partial derivatives, evaluated at the critical point.
        H = construct_hessian(f_xx, f_yy, f_xy, f_yx)
        lambda1, lambda2 = find_eigenvalues_2by2(H[0][0], H[0][1], H[1][0], H[1][1])    # Eigenvalues tell us the curvature of the function near that point.
        if lambda1 > 0 and lambda2 > 0:
            result = "Minimum"
        elif lambda1 < 0 and lambda2 < 0:
            result = "Maximum"
        elif lambda1 * lambda2 < 0:
            result = "Saddle Point"   # If eigen values have opposite signs.
        else:
            result = "Inconclusive"   # If any eigen value is zero or unclear.
        print(f"Critical point: {result}")
else:
    print("No critical points found.")