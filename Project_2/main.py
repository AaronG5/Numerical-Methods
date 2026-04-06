# 6: 6, 17

# A X = F
# L L^T = A
# L L^T X = F
# L^T X = Y
# L Y = F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
         L[k][j] = (A[j][k] - np.sum(L[j][:j] * L[k][:j])) / L[j][j]

      L[k][k] = np.sqrt(A[k][k] - np.sum(L[k][:k]**2)) # Find diagonal l

   return L

def solve_cholesky(L, F, N):
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

   start_L = time.time()
   L = create_L(A, N)
   end_L = time.time()

   time_cholesky = 0

   while True:
      F = create_F(X, N)

      start_cholesky = time.time()
      X_new = solve_cholesky(L, F, N)
      end_cholesky = time.time()
      time_cholesky += end_cholesky - start_cholesky

      if np.max(np.abs(X_new - X)) < epsilon:
         break

      X = X_new

   return X, end_L - start_L, time_cholesky

def solve_2():
   
   return 1

def plot_cholesky(df):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

   # Scaling factors
   const1 = df['Choleskio laik.'].iloc[-1] / df['N'].iloc[-1]**3
   const2 = df['Lygčių spr. laik.'].iloc[-1] / df['N'].iloc[-1]**2

   # Cholesky plot
   ax1.plot(df['N'], df['Choleskio laik.'], label='Išmatuotas laikas')
   ax1.plot(df['N'], df['N']**3 * const1, label='O(N³) teorinis', linestyle='--')
   ax1.set_xlabel('N')
   ax1.set_ylabel('Laikas (ms)')
   ax1.set_title('Choleskio dekompozicija')
   ax1.legend()
   ax1.grid(True)

   # Triangular plot
   ax2.plot(df['N'], df['Lygčių spr. laik.'], label='Išmatuotas laikas')
   ax2.plot(df['N'], df['N']**2 * const2, label='O(N²) teorinis', linestyle='--')
   ax2.set_xlabel('N')
   ax2.set_ylabel('Laikas (ms)')
   ax2.set_title('Trikampių lygčių sprendimas')
   ax2.legend()
   ax2.grid(True)

   plt.tight_layout()
   plt.savefig('Project_2/Cholesky.png', dpi=300)
   plt.show()

def main():
   N = np.linspace(4, 1000, 200, dtype=int)
   rows = []

   for N_instance in N:
      result, time_1, time_2 = solve_1(N_instance)
      rows.append({
         'N': N_instance,
         'Choleskio laik.': time_1 * 1000,
         'Lygčių spr. laik.': time_2 * 1000,
      })
   
   df = pd.DataFrame(rows)
   df.to_csv('Project_2/Cholesky.csv')

   plot_cholesky(df)

   return 1


if __name__ == '__main__':
   main()