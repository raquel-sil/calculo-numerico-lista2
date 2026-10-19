"""
Questão 2 - Sistema 3x3 resolvido pelos três métodos diretos:
  4x1 -  x2 +  x3 =  8
  2x1 + 5x2 + 2x3 =  3
   x1 + 2x2 + 4x3 = 11

Observação: o enunciado indica (-1, 1, 3), mas esse vetor não satisfaz o
sistema. A solução correta é (1, -1, 3), verificada abaixo.

Dependências: numpy
Execução:     python questao2_sistema_3x3.py
"""

import numpy as np


def cramer(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-12:
        raise ValueError("det(A) = 0: sistema sem solução única.")
    x = np.zeros(len(b))
    for i in range(len(b)):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    return x


def eliminacao_gauss(A, b):
    M = np.hstack([np.asarray(A, dtype=float),
                   np.asarray(b, dtype=float).reshape(-1, 1)])
    n = M.shape[0]
    for k in range(n - 1):
        p = k + np.argmax(np.abs(M[k:, k]))
        if abs(M[p, k]) < 1e-12:
            raise ValueError("Pivô nulo: sistema singular.")
        if p != k:
            M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            M[i, k:] -= m * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


def decomposicao_lu(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - L[k, :k] @ U[:k, j]
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]
    return L, U


def resolver_lu(A, b):
    L, U = decomposicao_lu(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


if __name__ == "__main__":
    A = np.array([[4.0, -1.0, 1.0],
                  [2.0, 5.0, 2.0],
                  [1.0, 2.0, 4.0]])
    b = np.array([8.0, 3.0, 11.0])

    print("Questão 2 - sistema 3x3\n")

    for nome, x in [("Cramer", cramer(A, b)),
                    ("Eliminação de Gauss", eliminacao_gauss(A, b)),
                    ("Decomposição LU", resolver_lu(A, b))]:
        res = np.linalg.norm(A @ x - b, np.inf)
        print(f"{nome:<22}: x = {np.round(x, 6)}   resíduo = {res:.2e}")

    print("\nVerificação de (-1, 1, 3) no sistema:", A @ np.array([-1, 1, 3]), "(esperado: [8, 3, 11])")
    print("Verificação de ( 1,-1, 3) no sistema:", A @ np.array([1, -1, 3]), "(esperado: [8, 3, 11])")  
