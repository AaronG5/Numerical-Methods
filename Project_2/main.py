# 6: 6, 17

# A X = F
# L L^T = A
# L L^T X = F
# L^T X = Y
# L Y = F
import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt
import time

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
   precision = 0.01 # Epsilon
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

   while True:
      r = A @ Z

      tau = (Z @ Z) / (r @ Z)
      
      X = X - tau * Z
      Z = Z - tau * r 

      if Z @ Z < precision**2:
         break

   return X

def solve_built_in(N):
   A = create_A(N)
   X = np.zeros(N)
   F = create_F(X, N)

   built_in_1 = np.linalg.solve(A, F) # 1

   built_in_2 = linalg.lu_solve(linalg.lu_factor(A), F) # 2

   built_in_3 = linalg.cho_solve(linalg.cho_factor(A), F) # 3

   func = lambda X: A @ X - create_F(X, N)
   built_in_4 = fsolve(func, np.zeros(N))

   return built_in_1, built_in_2, built_in_3, built_in_4

def plot_cholesky(df):
   # Scaling factors
   const1_1 = df['Choleskio laik.'].iloc[-1] / df['N'].iloc[-1]**3
   const1_2 = df['Choleskio laik.'].iloc[-1] / df['N'].iloc[-1]**2
   const2_1 = df['Lygčių spr. laik.'].iloc[-1] / df['N'].iloc[-1]**2
   const2_2 = df['Lygčių spr. laik.'].iloc[-1] / df['N'].iloc[-1]

   # Cholesky plot
   fig1, ax1 = plt.subplots(figsize=(7, 5))
   ax1.plot(df['N'], df['Choleskio laik.'], label='Išmatuotas laikas')
   ax1.plot(df['N'], df['N']**3 * const1_1, label='O(N³)', linestyle='--', color='red') 
   ax1.plot(df['N'], df['N']**2 * const1_2, label='O(N²)', linestyle='--', color='orange')
   ax1.set_xlabel('N')
   ax1.set_ylabel('Laikas (ms)')
   ax1.set_title('Choleskio dekompozicija')
   ax1.legend()
   ax1.grid(True)
   fig1.savefig('Project_2/Cholesky_decomp.png', dpi=300)

   # Triangular plot
   fig2, ax2 = plt.subplots(figsize=(7, 5))
   ax2.plot(df['N'], df['Lygčių spr. laik.'], label='Išmatuotas laikas')
   ax2.plot(df['N'], df['N']**2 * const2_1, label='O(N²)', linestyle='--', color='red')
   ax2.plot(df['N'], df['N'] * const2_2, label='O(N)', linestyle='--', color='orange') 
   ax2.set_xlabel('N')
   ax2.set_ylabel('Laikas (ms)')
   ax2.set_title('Trikampių lygčių sprendimas')
   ax2.legend()
   ax2.grid(True)
   fig2.savefig('Project_2/Cholesky_triangular.png', dpi=300)

   # plt.show()

def main():
   start = time.time()
   N = np.linspace(4, 1000, 200, dtype=int)
   rows = []

   for N_instance in N:
      _, time_1, time_2 = solve_cholesky(N_instance)
      rows.append({
         'N': N_instance,
         'Choleskio laik.': time_1 * 1000,
         'Lygčių spr. laik.': time_2 * 1000,
      })
   
   df = pd.DataFrame(rows)
   df.to_csv('Project_2/Cholesky.csv')

   plot_cholesky(df)

   M = 4
   X_ans_1, _, _ = solve_cholesky(M)                              # 1
   X_ans_2 = solve_steepest_descent(M)                            # 2
   X_ans_3_1, X_ans_3_2, X_ans_3_3, X_ans_3_4 = solve_built_in(M) # 3

   print('Cholesky method:', X_ans_1)
   print('Steepest descent method:', X_ans_2)
   print('Built-in method 1:', X_ans_3_1)
   print('Built-in method 2:', X_ans_3_2)
   print('Built-in method 3:', X_ans_3_3)
   print('Built-in method 4:', X_ans_3_4)

   end = time.time()

   print(f'Program took {(end - start)} s\n')

   return 1

if __name__ == '__main__':
   main()
