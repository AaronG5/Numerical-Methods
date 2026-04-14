import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
import time
from matrix import create_A, create_F, create_L

def solve_triangular(L, F, N):
   Y = np.zeros(N)
   X = np.zeros(N)

   for i in range(N): # L Y = F
      Y[i] = (F[i] - L[i][:i] @ Y[:i]) / L[i][i]
   for i in range(N-1, -1, -1): # L^T X = Y
      X[i] = (Y[i] - L.T[i][i+1:] @ X[i+1:]) / L[i][i]
   
   # True N^2 (no vectorization)
   # for i in range(N): # L Y = F
   #    s = 0
   #    for j in range(i):
   #       s += L[i][j] * Y[j]
   #    Y[i] = (F[i] - s) / L[i][i]
   # for i in range(N-1, -1, -1): # L^T X = Y
   #    s = 0
   #    for j in range(i+1, N):
   #       s += L.T[i][j] * X[j]
   #    X[i] = (Y[i] - s) / L[i][i]

   return X

def solve_cholesky(N):
   precision = 0.01
   A = create_A(N)
   X = np.zeros(N)

   start_L = time.time()
   L = create_L(A, N)
   end_L = time.time()

   time_cholesky = 0

   while True:
      F = create_F(X, N)

      start_cholesky = time.time()
      X_new = solve_triangular(L, F, N)
      end_cholesky = time.time()
      time_cholesky += end_cholesky - start_cholesky

      if np.max(np.abs(X_new - X)) < precision:
         break

      X = X_new

   return X, end_L - start_L, time_cholesky

def solve_steepest_descent(N):
   precision = 0.001
   A = create_A(N)
   X = np.zeros(N)
   F = create_F(X, N)
   Z = A @ X - F

   iterations = 0
   while True:
      r = A @ Z

      tau = (Z @ Z) / (r @ Z)
      
      X = X - tau * Z
      Z = Z - tau * r 
      iterations += 1

      # if Z @ Z < precision**2 * (F @ F): # Better, but takes much longer for all instances of N
      if Z @ Z < precision**2:
         break

   return X, iterations

def solve_built_in(N):
   A = create_A(N)
   X = np.zeros(N)
   F = create_F(X, N)

   built_in_results = []
   built_in_times = []

   start = time.time()
   built_in_results.append(np.linalg.solve(A, F))                       # 1
   built_in_times.append(time.time() - start)

   start = time.time()
   built_in_results.append(linalg.lu_solve(linalg.lu_factor(A), F))     # 2
   built_in_times.append(time.time() - start)
   
   start = time.time()
   built_in_results.append(linalg.cho_solve(linalg.cho_factor(A), F))   # 3
   built_in_times.append(time.time() - start)

   start = time.time()
   func = lambda X: A @ X - create_F(X, N)
   built_in_results.append(fsolve(func, np.zeros(N)))                   # 4
   built_in_times.append(time.time() - start)

   return built_in_results, built_in_times
