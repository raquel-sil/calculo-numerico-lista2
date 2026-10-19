"""
Questão 9 - Projeto: consumo de energia em uma rede elétrica

Parte 1: modelagem por método dos nós (Kirchhoff das correntes):
         sum_j g_ij (V_i - V_j) + g_terra V_i = I_i   ->   G V = I
Parte 2: sistema pequeno do enunciado, resolvido por Cramer, Gauss e LU
Parte 3: rede de 10 nós com Gauss, LU, Jacobi e Gauss-Seidel
Parte 4: iterações, tempo, resíduo, sensibilidade e condições de convergência

Dependências: numpy
Execução:     python questao9_rede_eletrica.py
"""

import time
import numpy as np


# ---------------- Métodos diretos ----------------

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


# ---------------- Métodos iterativos ----------------

def criterio_linhas(A):
    A = np.abs(np.asarray(A, dtype=float))
    return max((A[i].sum() - A[i, i]) / A[i, i] for i in range(len(A)))


def jacobi(A, b, x0, eps=1e-6, max_iter=10000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(1, max_iter + 1):
        x_novo = np.array([
            (b[i] - sum(A[i, j] * x[j] for j in range(n) if j != i)) / A[i, i]
            for i in range(n)
        ])
        erro = np.max(np.abs(x_novo - x))
        x = x_novo
        if erro < eps:
            return x, k
    raise RuntimeError("Jacobi não convergiu.")


def gauss_seidel(A, b, x0, eps=1e-6, max_iter=10000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(1, max_iter + 1):
        x_ant = x.copy()
        for i in range(n):
            soma1 = A[i, :i] @ x[:i]
            soma2 = A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - soma1 - soma2) / A[i, i]
        erro = np.max(np.abs(x - x_ant))
        if erro < eps:
            return x, k
    raise RuntimeError("Gauss-Seidel não convergiu.")


# ---------------- Modelo da rede ----------------

def montar_rede():
    """
    Rede resistiva com 10 nós. Anel 1-2-...-10-1 mais 3 atalhos.
    Cada nó está ligado à terra por g_terra = 0,5 S.
    Correntes injetadas I em amperes.
    """
    n = 10
    g_terra = 0.5
    arestas = [
        (1, 2, 2.0), (2, 3, 1.0), (3, 4, 2.0), (4, 5, 1.0), (5, 6, 2.0),
        (6, 7, 1.0), (7, 8, 2.0), (8, 9, 1.0), (9, 10, 2.0), (10, 1, 1.0),
        (1, 6, 0.5), (3, 8, 0.8), (2, 7, 0.6),
    ]

    G = np.zeros((n, n))
    for i in range(n):
        G[i, i] += g_terra
    for i, j, g in arestas:
        i, j = i - 1, j - 1
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    I = np.array([10.0, 0.0, -5.0, 0.0, 8.0, 0.0, 0.0, -3.0, 0.0, 2.0])
    return G, I


# ---------------- Execução ----------------

if __name__ == "__main__":
    print("=" * 70)
    print("Parte 2 - sistema pequeno do enunciado")
    print("=" * 70)
    A3 = np.array([[3.0, -1.0, 1.0],
                   [-2.0, 4.0, 0.0],
                   [1.0, 1.0, 5.0]])
    b3 = np.array([5.0, 6.0, -4.0])
    for nome, x in [("Cramer", cramer(A3, b3)),
                    ("Gauss", eliminacao_gauss(A3, b3)),
                    ("LU", resolver_lu(A3, b3))]:
        print(f"  {nome:<8}: {np.round(x, 6)}   resíduo = "
              f"{np.linalg.norm(A3 @ x - b3, np.inf):.2e}")

    print("\n" + "=" * 70)
    print("Parte 1 e 3 - rede de 10 nós")
    print("=" * 70)
    G, I = montar_rede()
    n = len(I)
    eps = 1e-6
    x0 = np.zeros(n)

    print("Matriz G (S):")
    print(G)
    print("Correntes I (A):", I)
    print(f"cond(G) = {np.linalg.cond(G):.4f}")
    print(f"Critério das linhas: max = {criterio_linhas(G):.4f} "
          f"-> convergente: {criterio_linhas(G) < 1}")

    resultados = []

    t0 = time.perf_counter()
    for _ in range(200):
        x_g = eliminacao_gauss(G, I)
    resultados.append(("Gauss", "-", (time.perf_counter() - t0) / 200, x_g))

    t0 = time.perf_counter()
    for _ in range(200):
        x_lu = resolver_lu(G, I)
    resultados.append(("LU", "-", (time.perf_counter() - t0) / 200, x_lu))

    t0 = time.perf_counter()
    x_j, it_j = jacobi(G, I, x0, eps=eps)
    resultados.append(("Jacobi", it_j, time.perf_counter() - t0, x_j))

    t0 = time.perf_counter()
    x_gs, it_gs = gauss_seidel(G, I, x0, eps=eps)
    resultados.append(("Gauss-Seidel", it_gs, time.perf_counter() - t0, x_gs))

    x_ref = np.linalg.solve(G, I)

    print(f"\n{'Método':<14}{'Iterações':>10}{'Tempo (ms)':>13}"
          f"{'Resíduo':>14}{'Erro vs ref':>14}")
    print("-" * 65)
    for nome, it, t, x in resultados:
        res = np.linalg.norm(G @ x - I, np.inf)
        erro = np.linalg.norm(x - x_ref, np.inf)
        print(f"{nome:<14}{str(it):>10}{t * 1000:>13.4f}{res:>14.2e}{erro:>14.2e}")

    print("\nTensões nodais (V):")
    for i, v in enumerate(x_ref, start=1):
        print(f"  nó {i:>2}: {v: .6f}")

    print("\n" + "=" * 70)
    print("Parte 4 - sensibilidade (perturbação de 1% em I)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    dI = I * 0.01 * rng.uniform(-1, 1, n)
    x_pert = eliminacao_gauss(G, I + dI)
    erro_rel_x = np.linalg.norm(x_pert - x_ref) / np.linalg.norm(x_ref)
    erro_rel_b = np.linalg.norm(dI) / np.linalg.norm(I)
    print(f"erro relativo em b: {erro_rel_b:.4e}")
    print(f"erro relativo em x: {erro_rel_x:.4e}")
    print(f"fator de amplificação: {erro_rel_x / erro_rel_b:.4f} "
          f"(limite: cond(G) = {np.linalg.cond(G):.4f})")
