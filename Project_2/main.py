# 6: 6, 17
import numpy as np

def solve_1():
   epsilon = 0.01 # Precision

   return 1

def main():
   N = 4 # Matrix size

   A = np.zeros((N, N), dtype=int)

   np.fill_diagonal(A, 2)
   np.fill_diagonal(A[1:], -1)
   np.fill_diagonal(A[:, 1:], -1)

   return


if __name__ == "__main__":
   main()