"""
Questão 1 - Métodos diretos para sistemas lineares Ax = b
  (i)   Regra de Cramer
  (ii)  Eliminação de Gauss (com pivoteamento parcial)
  (iii) Decomposição A = LU (Doolittle)
 
Dependências: numpy
Execução:     python questao1_metodos_diretos.py
"""
 
import numpy as np
 
 
def cramer(A, b):
    """
    Passo a passo:
      1. Calcula det(A). Se for zero, não há solução única.
      2. Para cada i, troca a coluna i de A pelo vetor b, obtendo A_i.
      3. x_i = det(A_i) / det(A).
    """
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
    """
    Passo a passo:
      1. Monta a matriz aumentada [A | b].
      2. Para cada coluna k, escolhe o maior |pivô| abaixo da diagonal e troca linhas.
      3. Zera os elementos abaixo do pivô: L_i <- L_i - m_ik * L_k, com m_ik = a_ik / a_kk.
      4. Resolve o sistema triangular superior por substituição retroativa.
    """
    M = np.hstack([np.asarray(A, dtype=float),
                   np.asarray(b, dtype=float).reshape(-1, 1)])
    n = M.shape[0]
 
    # Fase 1: eliminação progressiva
    for k in range(n - 1):
        p = k + np.argmax(np.abs(M[k:, k]))
        if abs(M[p, k]) < 1e-12:
            raise ValueError("Pivô nulo: sistema singular.")
        if p != k:
            M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            M[i, k:] -= m * M[k, k:]
 
    # Fase 2: substituição retroativa
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x
 
 
def decomposicao_lu(A):
    """
    Doolittle (L com diagonal unitária), para k = 1..n:
      U_kj = a_kj - sum_{p<k} L_kp U_pj            (j >= k)
      L_ik = (a_ik - sum_{p<k} L_ip U_pk) / U_kk   (i > k)
    """
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
    """Resolve Ly = b (progressiva) e depois Ux = y (retroativa)."""
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
 
 
def imprimir_solucao(nome, x, A, b):
    partes = ", ".join(f"x{i+1} = {v: .6f}" for i, v in enumerate(x))
    res = np.linalg.norm(A @ x - b, np.inf)
    print(f"  {nome:<22}: {partes}   | resíduo = {res:.2e}")
 
 
if __name__ == "__main__":
    # Sistema da Questão 2, usado para testar os três métodos
    A = np.array([[4.0, -1.0, 1.0],
                  [2.0, 5.0, 2.0],
                  [1.0, 2.0, 4.0]])
    b = np.array([8.0, 3.0, 11.0])
 
    print("Questão 1 - métodos diretos aplicados ao sistema da Questão 2\n")
    imprimir_solucao("Cramer", cramer(A, b), A, b)
    imprimir_solucao("Eliminação de Gauss", eliminacao_gauss(A, b), A, b)
    imprimir_solucao("Decomposição LU", resolver_lu(A, b), A, b)
 
    L, U = decomposicao_lu(A)
    print("\nL =\n", L)
    print("\nU =\n", U)
 
