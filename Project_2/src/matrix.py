import numpy as np

def create_A(N):
   A = np.zeros((N, N), dtype=float)

   np.fill_diagonal(A, 2)
   np.fill_diagonal(A[1:], -1)
   np.fill_diagonal(A[:, 1:], -1)

   return A

def create_F(X, N):
   c = 1 / (N + 1)**2
   return c * (X**2 + 1)

def create_L(A, N):
   L = np.zeros((N, N))
   L[0][0] = np.sqrt(A[0][0])
   
   for k in range(N):
      for j in range(k):
         L[k][j] = (A[j][k] - L[j][:j] @ L[k][:j]) / L[j][j]
      L[k][k] = np.sqrt(A[k][k] - np.sum(L[k][:k]**2)) # Find diagonal l

   # True N^3 (no vectorization)
   # for k in range(N):
   #    for j in range(k):
   #       s = 0
   #       for m in range(j):
   #          s += L[k][m] * L[j][m]
   #       L[k][j] = (A[k][j] - s) / L[j][j]

   #    s = 0
   #    for m in range(k):
   #       s += L[k][m]**2
   #    L[k][k] = np.sqrt(A[k][k] - s)

   return L