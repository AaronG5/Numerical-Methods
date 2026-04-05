# 6: 6, 17

# A X = F
# L L^T = A
# L L^T X = F
# L^T X = Y
# L Y = F
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
         L[k][j] = (A[j][k] - np.sum(L[j][:j] * L[k][:j])) / L[j][j]

      L[k][k] = np.sqrt(A[k][k] - np.sum(L[k][:k]**2)) # Find diagonal l

   return L

def solve_cholesky(A, F, N):
   L = create_L(A, N)
   Y = np.zeros(N)
   X = np.zeros(N)

   for i in range(N): # L Y = F
      Y[i] = (F[i] - np.sum(L[i][:i] * Y[:i])) / L[i][i]

   for i in range(N-1, -1, -1): # L^T X = Y
      X[i] = (Y[i] - np.sum(L.T[i][i+1:] * X[i+1:])) / L[i][i]

   return X

def solve_1(N):
   epsilon = 0.01 # Precision
   A = create_A(N)
   X = np.zeros(N)

   while True:
      F = create_F(X, N)

      X_new = solve_cholesky(A, F, N)

      if np.max(np.abs(X_new - X)) < epsilon:
         break

      X = X_new

   return X

def main():
   N = 4 # Matrix size N x N
   A = create_A(N)

   result = solve_1(N)

   print("Residual:", np.max(np.abs(A @ result - create_F(result, N))))
   print(result)

   return 1


if __name__ == "__main__":
   main()