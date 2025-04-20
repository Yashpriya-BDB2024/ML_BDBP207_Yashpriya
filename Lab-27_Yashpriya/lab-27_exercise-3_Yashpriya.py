### Q-3: Determine the concavity of f(x, y) = x^3 + 2y^3 - xy at (i) (0,0) (ii) (3, 3), (iii) (3, -3).

from lab_27_excerise_2_Yashpriya import find_eigenvalues_2by2
import sympy as sp

"""
About Concavity:
It tells us the shape or curvature of a function — whether it bends upward or downward.
1. If eigenvalues of Hessian are positive, then local minimum (concave up).
2. If eigenvalues of Hessian are negative, then local maximum (concave down).
3. If mixed signs, then it's a saddle point (change in curvature).
"""

def compute_second_derivatives(f_expr, x_value, y_value):
    x, y = sp.symbols('x y')
    f_xx = sp.diff(f_expr, x, x)   # 2nd partial derivative of function f_expr w.r.t. x
    f_yy = sp.diff(f_expr, y, y)   # 2nd partial derivative of function f_expr w.r.t. y
    f_xy = sp.diff(f_expr, x, y)   # mixed partial derivative of function f_expr w.r.t. x and y
    f_yx = sp.diff(f_expr, y, x)   # mixed partial derivative of function f_expr w.r.t. y and x
    # Evaluate second order partial derivatives at the given point (x_value, y_value)
    f_xx_at_point = f_xx.subs({x: x_value, y: y_value})
    f_yy_at_point = f_yy.subs({x: x_value, y: y_value})
    f_xy_at_point = f_xy.subs({x: x_value, y: y_value})
    f_yx_at_point = f_yx.subs({x: x_value, y: y_value})
    return f_xx_at_point, f_yy_at_point, f_xy_at_point, f_yx_at_point  # These will be used to construct the Hessian matrix.

def construct_hessian(f_xx, f_yy, f_xy, f_yx):   # Takes the second derivatives as input.
    H = [[f_xx, f_xy],
         [f_yx, f_yy]]
    return H  # Returns the Hessian matrix.

def determine_concavity_at_points(H):
    eigenvalues = find_eigenvalues_2by2(H[0][0], H[0][1], H[1][0], H[1][1])   # It passes the individual matrix elements to the function.
    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        return "Concave up"
    elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
        return "Concave down"
    else:
        return "Saddle point (indefinite concavity)"

def check_concavity(f_expr, points):   # Takes the function expression (f_expr) and a list of points to evaluate.
    for point in points:   # This loops over each point in the points list.
        x_value, y_value = point   # Each point is a tuple.
        f_xx, f_yy, f_xy, f_yx = compute_second_derivatives(f_expr, x_value, y_value)   # to get the second derivatives of f_expr at that point.
        H = construct_hessian(f_xx, f_yy, f_xy, f_yx)
        concavity = determine_concavity_at_points(H)
        print(f"At point {point}:")
        print(f"Concavity: {concavity}")

def main():
    x, y = sp.symbols('x y')   # This creates symbolic variables x and y to define the function.
    f = x**3 + 2*y**3 + x*y
    points = [(0,0), (3,3), (3,-3)]  # Points to evaluate
    check_concavity(f, points)

if __name__ == "__main__":
    main()