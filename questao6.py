"""
Questão 6 - Método de Gauss-Seidel
  5x1 +  x2 +  x3 = 5
  3x1 + 4x2 +  x3 = 6
  3x1 + 3x2 + 6x3 = 0

x0 = (0, 0, 0), tolerância ε < 10^-2
Solução de referência do enunciado: (1, 1, -1)

Gauss-Seidel: x_i^(k+1) = (b_i - sum_{j<i} a_ij x_j^(k+1) - sum_{j>i} a_ij x_j^(k)) / a_ii
Parada:       ||x^(k+1) - x^(k)||_inf < ε

Dependências: numpy
Execução:     python questao6_gauss_seidel.py
"""

import numpy as np


def criterio_linhas(A):
    """max_i sum_{j!=i} |a_ij| / |a_ii|. Converge se < 1."""
    A = np.abs(np.asarray(A, dtype=float))
    return max((A[i].sum() - A[i, i]) / A[i, i] for i in range(len(A)))


def criterio_sassenfeld(A):
    """
    beta_i = (sum_{j<i} |a_ij| beta_j + sum_{j>i} |a_ij|) / |a_ii|
    Converge se max_i beta_i < 1.
    """
    A = np.abs(np.asarray(A, dtype=float))
    n = A.shape[0]
    beta = np.zeros(n)
    for i in range(n):
        soma_antes = sum(A[i, j] * beta[j] for j in range(i))
        soma_depois = sum(A[i, j] for j in range(i + 1, n))
        beta[i] = (soma_antes + soma_depois) / A[i, i]
    return beta


def gauss_seidel(A, b, x0, eps=1e-2, max_iter=1000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.asarray(x0, dtype=float).copy()
    historico = []

    for k in range(1, max_iter + 1):
        x_ant = x.copy()
        for i in range(n):
            soma1 = A[i, :i] @ x[:i]          # já atualizados nesta iteração
            soma2 = A[i, i + 1:] @ x[i + 1:]  # ainda da iteração anterior
            x[i] = (b[i] - soma1 - soma2) / A[i, i]
        erro = np.max(np.abs(x - x_ant))
        historico.append((k, x.copy(), erro))
        if erro < eps:
            return x, k, historico
    raise RuntimeError("Gauss-Seidel não convergiu.")


if __name__ == "__main__":
    A = np.array([[5.0, 1.0, 1.0],
                  [3.0, 4.0, 1.0],
                  [3.0, 3.0, 6.0]])
    b = np.array([5.0, 6.0, 0.0])
    x0 = np.zeros(3)
    eps = 1e-2

    print("Questão 6 - Gauss-Seidel\n")
    beta = criterio_sassenfeld(A)
    print(f"Critério das linhas: max = {criterio_linhas(A):.4f} "
          f"(não garante, pois não é < 1)")
    print(f"Critério de Sassenfeld: betas = {beta}, max = {beta.max():.4f} "
          f"-> convergente: {beta.max() < 1}\n")

    x, k, historico = gauss_seidel(A, b, x0, eps=eps)

    print(f"{'k':>3} | {'x1':>11} {'x2':>11} {'x3':>11} | {'erro':>11}")
    print("-" * 56)
    for it, xi, erro in historico:
        print(f"{it:>3} | {xi[0]:>11.6f} {xi[1]:>11.6f} {xi[2]:>11.6f} | {erro:>11.3e}")

    print(f"\nIterações: {k} (ε = {eps})")
    print(f"Solução aproximada: {np.round(x, 6)}")
    print(f"Solução de referência: [ 1.  1. -1.]")
    print(f"Resíduo ||Ax - b||_inf = {np.linalg.norm(A @ x - b, np.inf):.2e}")
