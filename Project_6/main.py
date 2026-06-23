# 1. Netiesinės ir tiesinės lygtys
#     2. Netiesinių lygčių sistema

from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import os

from simple_iteration import simple_iteration, easy_system, my_hard_system
from newton import newton_method, jacobian_easy, jacobian_hard
from separation import separation_method
from plot import plot_diffs

def easy_system_original(X):
   x = X[0]
   y = X[1]
   f1_res = x - 0.125 * (8 * x - 4 * x**2 + y**2 + 1) # 0.5x^2 - 0.125y^2 - 0.125
   f2_res = y - 0.25 * (2 * x - x**2 + 4 * y - y**2 + 3) # 0.25x^2 - 0.5x + 0.25y^2 - 0.75
   return (f1_res, f2_res)

def hard_system_original(X):
   x = X[0]
   y = X[1]
   z = X[2]
   f1_res = x**2 + y**2 + z**2
   f2_res = x * y + x * z - z - 0.5 * y
   f3_res = x**2 * y * z - y**2 - 3 * z**3 * y - (1/3)
   return (f1_res, f2_res, f3_res)

def my_hard_system_original(X):
   x = X[0]
   y = X[1]
   z = X[2]
   f1_res = x**2 + y**2 + z**2 - 3
   f2_res = x * y + x * z - z - 0.5 * y
   f3_res = x**2 * y * z + y**2 - 3 * z**2 * y - (1/3)
   return (f1_res, f2_res, f3_res)

def test_method_easy(method, method_name, system, jacobian):
   valid_starting_points = []
   for i in range(-50, 50):
      i = i / 10
      for j in range(-50, 50):
         j = j / 10
         X = np.array((i, j))
         try:
            if method_name == 'Simple_iteration':
               res, _ = method(jacobian, X) # modified system for simple iteration instead of jacobian
            elif np.all(X == 0):
               continue
            else:
               res, _ = method(system, jacobian, X)

            if np.linalg.norm(system(res), np.inf) < 1e-6:
               valid_starting_points.append({
                  'x_val': i,
                  'y_val': j,
                  'x_res': res[0],
                  'y_res': res[1]
               })
         except np.linalg.LinAlgError:
            pass
   
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   filepath = os.path.join(result_dir, f'{method_name}_Easy_Valid_Start_Points.csv')
   df = pd.DataFrame(valid_starting_points)
   df.to_csv(filepath)

def test_method_hard(method, method_name, system, jacobian):
   unique_roots = set()
   valid_starting_points = []
   for i in range(-50, 50):
      i = i / 10
      for j in range(-50, 50):
         j = j / 10
         for k in range(-50, 50):
            k = k / 10
            X = np.array((i, j, k))
            try:
               if method_name == 'Simple_iteration':
                  res, _ = method(jacobian, X) # modified system for simple iteration instead of jacobian
               elif np.all(X == 0):
                  continue
               else:
                  res, _ = method(system, jacobian, X)

               if np.linalg.norm(system(res), np.inf) < 1e-6:
                  root = tuple(np.round(res, 5))

                  if root not in unique_roots:
                     unique_roots.add(root)
                     valid_starting_points.append({
                        'x_val': i,
                        'y_val': j,
                        'z_val': k,
                        'x_res': res[0],
                        'y_res': res[1],
                        'z_res': res[2]
                     })
            except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
               pass
   
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   filepath = os.path.join(result_dir, f'{method_name}_Hard_Valid_Start_Points.csv')
   df = pd.DataFrame(valid_starting_points)
   df.to_csv(filepath)

def main():
   X_easy = np.array((2.0, -1.0))
   X_hard = np.array((0.66, 1.43, 0.71))

   # Testing
   # test_method_easy(simple_iteration, 'Simple_iteration', easy_system_original, easy_system)
   # test_method_easy(newton_method, 'Newton', easy_system_original, jacobian_easy)
   # test_method_easy(separation_method, 'Separation', easy_system_original, jacobian_easy)

   # test_method_hard(simple_iteration, 'Simple_iteration', my_hard_system_original, my_hard_system)
   # test_method_hard(newton_method, 'Newton', my_hard_system_original, jacobian_hard)
   # test_method_hard(separation_method, 'Separation', my_hard_system_original, jacobian_hard)


   # Simple iteration method
   res_simple_iter_easy, iter_count_simple, simple_norms = simple_iteration(easy_system, X_easy)
   print(f'Simple iteration easy result: {res_simple_iter_easy} \
         \nPlugging values into system: {np.round(easy_system_original(res_simple_iter_easy), 5)} \
         \nIterations: {iter_count_simple + 1}\n')
   
   res_simple_iter_hard, iter_count_simple, _ = simple_iteration(my_hard_system, X_hard)
   print(f'Simple iteration hard result: {res_simple_iter_hard} \
         \nPlugging values into system: {np.round(my_hard_system_original(res_simple_iter_hard), 5)} \
         \nIterations: {iter_count_simple + 1}\n')

   # Newton method
   res_newton_easy, iter_count_newton, newton_norms = newton_method(easy_system_original, jacobian_easy, X_easy)
   print(f'Newton\'s easy result: {res_newton_easy} \
         \nPlugging values into system: {np.round(easy_system_original(res_newton_easy), 5)} \
         \nIterations: {iter_count_newton + 1}\n')
   
   res_newton_hard, iter_count_newton, _ = newton_method(my_hard_system_original, jacobian_hard, X_hard)
   print(f'Newton\'s hard result: {res_newton_hard} \
         \nPlugging values into system: {np.round(my_hard_system_original(res_newton_hard), 5)} \
         \nIterations: {iter_count_newton + 1}\n')
   
   # Separation method
   res_separation_easy, iter_count_separation, separation_norms = separation_method(easy_system_original, jacobian_easy, X_easy)
   print(f'Separation easy result: {res_separation_easy} \
         \nPlugging values into system: {np.round(easy_system_original(res_separation_easy), 5)} \
         \nIterations: {iter_count_separation + 1}\n')
   
   res_separation_hard, iter_count_separation, _ = separation_method(my_hard_system_original, jacobian_hard, X_hard)
   print(f'Separation hard result: {res_separation_hard} \
         \nPlugging values into system: {np.round(my_hard_system_original(res_separation_hard), 5)} \
         \nIterations: {iter_count_separation + 1}\n')

   # Scipy
   res_scipy_easy = fsolve(easy_system_original, X_easy)
   print(f'Fsolve easy result: {res_scipy_easy} \
         \nPlugging values into system: {np.round(easy_system_original(res_scipy_easy), 5)}\n')
   res_scipy_hard = fsolve(my_hard_system_original, X_hard)
   print(f'Fsolve hard result: {res_scipy_hard} \
         \nPlugging values into system: {np.round(my_hard_system_original(res_scipy_hard), 5)}\n')

   plot_diffs(simple_norms, newton_norms, separation_norms)

if __name__ == '__main__':
   main()