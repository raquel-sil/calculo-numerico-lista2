"""
Questão 7 - Polinômio interpolador de grau 4 que passa por
  (0, -1), (1, 1), (3, 3), (5, 2), (6, -2)

Método: y(x) = a4 x^4 + a3 x^3 + a2 x^2 + a1 x + a0 impõe 5 condições,
gerando um sistema V a = y com a matriz de Vandermonde V.
Resolvido por eliminação de Gauss.

Dependências: numpy
Execução:     python questao7_interpolacao.py
"""

import numpy as np


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


if __name__ == "__main__":
    xs = np.array([0.0, 1.0, 3.0, 5.0, 6.0])
    ys = np.array([-1.0, 1.0, 3.0, 2.0, -2.0])

    V = np.vander(xs, N=5)              # colunas: x^4, x^3, x^2, x, 1
    coef = eliminacao_gauss(V, ys)      # a4, a3, a2, a1, a0

    print("Questão 7 - polinômio interpolador de grau 4\n")
    print("Coeficientes (de x^4 até x^0):", coef)
    print("Diferença para numpy.linalg.solve:",
          np.max(np.abs(coef - np.linalg.solve(V, ys))))

    termos = []
    for c, g in zip(coef, [4, 3, 2, 1, 0]):
        if abs(c) < 1e-12:
            continue
        sufixo = f"x^{g}" if g > 1 else ("x" if g == 1 else "")
        termos.append(f"{c:+.6f}{sufixo}")
    print("\ny(x) = " + " ".join(termos))

    print("\nVerificação nos pontos dados:")
    for x_i, y_i, y_c in zip(xs, ys, np.polyval(coef, xs)):
        print(f"  x = {x_i:4.1f}   y_dado = {y_i:6.2f}   y_polinômio = {y_c:10.6f}")
