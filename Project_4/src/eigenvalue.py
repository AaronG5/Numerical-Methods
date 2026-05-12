# 1
import numpy as np
from matrix import create_A, create_B, create_jacobian_matrix
import time

def find_omega(eigen) -> float:
   return 2.0 / (1.0 + np.sqrt(1.0 - eigen**2))

def normalize(Y) -> np.ndarray:
   return Y / np.linalg.norm(Y, 2)

def find_eigenvalue(A) -> float:
   iter_count = 0
   N = len(A)
   eps = 0.001
   
   X1 = np.ones(N)
   lam = 0.0

   while True:
      X0 = X1.copy()
      lam_prev = lam
      X1 = normalize(A @ X0)
      lam = (X1 @ A @ X1) / (X1 @ X1)
      iter_count += 1

      if abs(lam - lam_prev) < eps:
         break

   return lam, iter_count

def relaxation_method(A, B, eps, X1=None, omega=None) -> np.ndarray:
   eigen_iter_count = None
   if omega is None:
      J = create_jacobian_matrix(A)
      eigenvalue, eigen_iter_count = find_eigenvalue(J)
      omega = find_omega(eigenvalue)
   
   iter_count = 0
   N = len(B)
   if X1 is None:
      X1 = np.zeros(N)
   D = np.diag(A)
   A = A.copy() - np.diag(D)
   
   while True:
      X0 = X1.copy()

      for i in range(N):
         X1[i] = (B[i] - A[i] @ X1) / D[i]
         X1[i] = (1.0 - omega) * X0[i] + omega * X1[i]

      iter_count += 1

      if np.linalg.norm(X1 - X0, 2) < eps:
         break
   if eigen_iter_count is None:
      return X1, iter_count
   return X1, iter_count, eigen_iter_count

def alternative_relaxation_method(A, B, eps) -> np.ndarray:
   jacobian_iter_count = 2
   N = len(B)
   D = np.diag(A)
   A_no_D = A.copy() - np.diag(D)
   
   X0 = np.zeros(N)
   X1 = (B - A_no_D @ X0) / D
   X2 = (B - A_no_D @ X1) / D
   
   lam_prev = 0.0
   lam = 0.0

   while True:
      lam = np.linalg.norm(X2 - X1, 2) / np.linalg.norm(X1 - X0, 2)


      if abs(lam - lam_prev) < eps:
         break
      
      lam_prev = lam
      X0 = X1.copy()
      X1 = X2.copy()
      X2 = (B - A_no_D @ X1) / D
      jacobian_iter_count += 1

   omega = find_omega(lam)

   res, relax_iter_count = relaxation_method(A, B, eps, X2.copy(), omega)  

   return res, jacobian_iter_count, relax_iter_count

def main():
   eps = 0.0001
   N = 5
   A = create_A(N)
   B = create_B(np.zeros(N))

   start = time.time()
   res, relaxation_iter_count, eigen_iter_count = relaxation_method(A, B, eps)
   end = time.time()
   time_1 = end - start
   print('\nRezultatai.\n' \
         '1. Relaksacijos metodas.\n' \
         f' - Sprendinio įvertis: {res}\n' \
         f' - Laipsnių metodo iteracijų skaičius: {eigen_iter_count}\n' \
         f' - Relaksacijos metodo iteracijų skaičius: {relaxation_iter_count}\n' \
         f' - Darbo laikas: {time_1}\n')
   
   start = time.time()
   res, jacobian_iter_count, relaxation_iter_count = alternative_relaxation_method(A, B, eps)
   end = time.time()
   time_2 = end - start
   print('\nRezultatai.\n' \
         '2. Alternatyvus relaksacijos metodas (be laipsnių metodo).\n' \
         f' - Sprendinio įvertis: {res}\n' \
         f' - Jakobio metodo iteracijų skaičius: {jacobian_iter_count}\n' \
         f' - Relaksacijos metodo iteracijų skaičius: {relaxation_iter_count}\n' \
         f' - Darbo laikas: {time_2}\n')
   
   print(f'\nMetodu veikimo laiko skirtumas: {abs(time_1 - time_2)}\n')

   return 0

if __name__ == '__main__':
   main()