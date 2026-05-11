import numpy as np

def create_A(N):
   A = np.zeros((N, N), dtype=float)

   np.fill_diagonal(A, 2)
   np.fill_diagonal(A[1:], -1)
   np.fill_diagonal(A[:, 1:], -1)

   return A

def create_B(X):
   N = len(X)
   c = 1 / (N + 1)**2
   return c * (X**2 + 1)

def create_jacobian_matrix(A) -> np.ndarray:
   D = np.diag(A)
   D_inv = np.diag(1.0 / D)
   LR = A - np.diag(D)
   return -D_inv @ LR