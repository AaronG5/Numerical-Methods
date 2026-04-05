# 6: 6, 17
import numpy as np

def f(x, c):
   return c * (x**2 + 1)

def create_A(N):
   A = np.zeros((N, N), dtype=float)

   np.fill_diagonal(A, 2)
   np.fill_diagonal(A[1:], -1)
   np.fill_diagonal(A[:, 1:], -1)

   return A

def create_F(N, X):
   c = 1 / (N + 1)**2
   return c * (X**2 + 1)

def create_L(A):
   N = len(A[0])
   L = np.zeros((N, N))
   L[k][k] = np.sqrt(A[k][k])
   
   for k in range(N):
      for j in range(k):
         L[k][j] = (A[j][k] - np.sum(L[j] * L[k])) / L[j][j]

      L[k][k] = np.sqrt(A[k][k] - np.sum(L[k]**2)) # Find diagonal l

   return L

def solve_cholesky(A, F):
   L = create_L(A)
   return 1

def solve_1(N):
   epsilon = 0.01 # Precision
   A = create_A(N)
   X = np.zeros(N)

   while True:
      F = create_F(N, X)

      X_new = solve_cholesky(A, F)

      if np.max(np.abs(X_new - X)) < epsilon:
         break

      X = X_new

   return X

def main():
   N = 4 # Matrix size

   solve_1(N)

   return 1


if __name__ == "__main__":
   main()