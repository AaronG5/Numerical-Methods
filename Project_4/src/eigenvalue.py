# 1
import numpy as np
from matrix import create_A, create_B

def find_omega(eigen) -> float:
   return 2.0 / (1.0 + np.sqrt(1.0 - eigen**2))

def normalize(Y) -> np.ndarray:
   return Y / np.linalg.norm(Y, 2)

def find_eigenvalue(A) -> float:
   N = len(A)
   eps = 0.001
   
   X1 = np.ones(N)
   lam = 0.0

   while True:
      X0 = X1.copy()
      lam_prev = lam
      X1 = normalize(A @ X0)
      lam = (X1 @ A @ X1) / (X1 @ X1)

      if abs(lam - lam_prev) < eps:
         break

   return lam

def relax_method(A, B, omega, eps) -> np.ndarray:
   N = len(B)
   X1 = np.zeros(N)
   D = np.diag(A)
   A = A.copy() - np.diag(D)
   
   while True:
      X0 = X1.copy()

      for i in range(N):
         X1[i] = (B[i] - A[i] @ X1) / D[i]
         X1[i] = (1.0 - omega) * X0[i] + omega * X1[i]

      if np.linalg.norm(X1 - X0, 2) < eps:
         break

   return X1

def create_jacobian_matrix(A) -> np.ndarray:
   D = np.diag(A)
   D_inv = np.diag(1.0 / D)
   LR = A - np.diag(D)
   return -D_inv @ LR

def main():
   eps = 0.0001
   N = 5
   A = create_A(N)
   B = create_B(np.zeros(N))

   J = create_jacobian_matrix(A)

   eigenvalue = find_eigenvalue(J)

   omega = find_omega(eigenvalue)

   res = relax_method(A, B, omega, eps)
   print(res)

   return 0

if __name__ == '__main__':
   main()