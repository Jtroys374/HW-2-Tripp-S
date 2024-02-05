import numpy as np


def GaussSeidel(Aaug, x, Niter=15):
    N = len(x)
    for _ in range(Niter):
        for i in range(N):
            sigma = 0
            for j in range(N):
                if j != i:
                    sigma += Aaug[i, j] * x[j]
            x[i] = (Aaug[i, -1] - sigma) / Aaug[i, i]
    return x


def main():
    # First set of linear equations
    Aaug1 = np.array([[3, 1, -1, 2],
                      [1, 4, 1, 12],
                      [2, 1, 2, 10]], dtype=float)
    x1 = np.zeros(3)
    x1 = GaussSeidel(Aaug1, x1)
    print("Solution for [3 1 -1, 1 4 1, 2 1 2] [x1, x2, x3] = [2, 12, 10]:")
    print(x1)

    # Second set of linear equations
    Aaug2 = np.array([[1, -10, 2, 4, 2],
                      [3, 1, 4, 12, 12],
                      [9, 2, 3, 4, 21],
                      [-1, 2, 7, 3, 37]], dtype=float)
    x2 = np.zeros(4)
    x2 = GaussSeidel(Aaug2, x2)
    print("\nSolution for [1 -10 2 4, 3 1 4 12, 9 2 3 4, -1 2 7 3] [x1, x2, x3, x4] = [2, 12, 21, 37]:")
    print(x2)


if __name__ == "__main__":
    main()
