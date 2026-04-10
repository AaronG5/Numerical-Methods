# 6: 6, 17

# A X = F
# L L^T = A
# L L^T X = F
# L^T X = Y
# L Y = F
import numpy as np
import pandas as pd
import time
import os
from solvers import solve_cholesky, solve_steepest_descent, solve_built_in
from plotters import plot_cholesky, plot_steepest_descent

def main():
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
   os.makedirs(result_dir, exist_ok=True)

   start = time.time()
   N = np.linspace(4, 1000, 200, dtype=int)
   rows = []

   for N_instance in N:
      _, time_cholesky, time_triangular = solve_cholesky(N_instance)
      _, time_steepest_descent = solve_steepest_descent(N_instance)
      _, built_in_times = solve_built_in(N_instance)

      rows.append({
         'N': N_instance,
         'Choleskio laik.': time_cholesky * 1000,
         'Lygčių spr. laik.': time_triangular * 1000,
         'Didžiausio nuolydžio laik.': time_steepest_descent * 1000,
         'Python 1-os funkcijos laik.': built_in_times[0] * 1000,
         'Python 2-os funkcijos laik.': built_in_times[1] * 1000, 
         'Python 3-os funkcijos laik.': built_in_times[2] * 1000, 
         'Python 4-os funkcijos laik.': built_in_times[3] * 1000, 
      })
   
   table_filepath = os.path.join(result_dir, 'Laikai.csv')
   df = pd.DataFrame(rows)
   df.to_csv(table_filepath)

   plot_cholesky(df, result_dir)
   plot_steepest_descent(df, result_dir)

   M = 4
   X_res_1, _, _ = solve_cholesky(M)      # 1
   X_res_2, _ = solve_steepest_descent(M) # 2
   X_results, _ = solve_built_in(M)       # 3

   print('Cholesky method:', X_res_1)
   print('Steepest descent method:', X_res_2)
   print('Built-in method 1:', X_results[0])
   print('Built-in method 2:', X_results[1])
   print('Built-in method 3:', X_results[2])
   print('Built-in method 4:', X_results[3])

   end = time.time()

   print(f'Program took {(end - start)} s\n')

   return 1

if __name__ == '__main__':
   main()
