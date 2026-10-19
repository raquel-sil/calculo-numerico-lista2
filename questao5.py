"""
Questão 5 - Método de Jacobi
  10x1 + 2x2 +  x3 = 7
   x1 + 5x2 +  x3 = -8
  2x1 + 3x2 + 10x3 = 6

x0 = (0,7; -1,6; 0,6), tolerância ε < 10^-2
Solução de referência do enunciado: (1, -2, 1)

Jacobi:      x_i^(k+1) = (b_i - sum_{j!=i} a_ij x_j^(k)) / a_ii
Parada:      ||x^(k+1) - x^(k)||_inf < ε

Dependências: numpy
Execução:     python questao5_jacobi.py
"""

import numpy as np


def criterio_linhas(A):
    """Retorna max_i sum_{j!=i} |a_ij| / |a_ii|. Converge se < 1."""
    A = np.abs(np.asarray(A, dtype=float))
    razoes = [(A[i].sum() - A[i, i]) / A[i, i] for i in range(len(A))]
    return max(razoes)


def jacobi(A, b, x0, eps=1e-2, max_iter=1000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.asarray(x0, dtype=float).copy()
    historico = []

    for k in range(1, max_iter + 1):
        x_novo = np.array([
            (b[i] - sum(A[i, j] * x[j] for j in range(n) if j != i)) / A[i, i]
            for i in range(n)
        ])
        erro = np.max(np.abs(x_novo - x))
        x = x_novo
        historico.append((k, x.copy(), erro))
        if erro < eps:
            return x, k, historico
    raise RuntimeError("Jacobi não convergiu.")


if __name__ == "__main__":
    A = np.array([[10.0, 2.0, 1.0],
                  [1.0, 5.0, 1.0],
                  [2.0, 3.0, 10.0]])
    b = np.array([7.0, -8.0, 6.0])
    x0 = np.array([0.7, -1.6, 0.6])
    eps = 1e-2

    print("Questão 5 - Jacobi\n")
    print(f"Critério das linhas: max = {criterio_linhas(A):.4f}  "
          f"-> convergente: {criterio_linhas(A) < 1}\n")

    x, k, historico = jacobi(A, b, x0, eps=eps)

    print(f"{'k':>3} | {'x1':>11} {'x2':>11} {'x3':>11} | {'erro':>11}")
    print("-" * 56)
    for it, xi, erro in historico:
        print(f"{it:>3} | {xi[0]:>11.6f} {xi[1]:>11.6f} {xi[2]:>11.6f} | {erro:>11.3e}")

    print(f"\nIterações: {k} (ε = {eps})")
    print(f"Solução aproximada: {np.round(x, 6)}")
    print(f"Solução de referência: [ 1. -2.  1.]")
    print(f"Resíduo ||Ax - b||_inf = {np.linalg.norm(A @ x - b, np.inf):.2e}")
