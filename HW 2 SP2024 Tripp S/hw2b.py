import numpy as np


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    x_prev = x0
    x_curr = x1
    iter_count = 0

    while iter_count < maxiter:
        f_prev = fcn(x_prev)
        f_curr = fcn(x_curr)
        if abs(x_curr - x_prev) < xtol:
            return x_curr
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        x_prev = x_curr
        x_curr = x_next
        iter_count += 1

    return x_curr


def main():
    # First equation: x - 3*cos(x) = 0
    fcn1 = lambda x: x - 3 * np.cos(x)
    x0_1, x1_1 = 1, 2
    root1 = Secant(fcn1, x0_1, x1_1, maxiter=5, xtol=1e-4)
    print(f"Root of x - 3*cos(x) = 0: {root1:.6f}")

    # Second equation: cos(2*x) * x^3 = 0
    fcn2 = lambda x: np.cos(2 * x) * x ** 3
    x0_2, x1_2 = 1, 2
    root2 = Secant(fcn2, x0_2, x1_2, maxiter=15, xtol=1e-8)
    print(f"Root of cos(2*x) * x^3 = 0: {root2:.6f}")

    # Third equation: cos(2*x) * x^3 = 0 (with fewer iterations)
    root3 = Secant(fcn2, x0_2, x1_2, maxiter=3, xtol=1e-8)
    print(f"Root of cos(2*x) * x^3 = 0 (fewer iterations): {root3:.6f}")


if __name__ == "__main__":
    main()
