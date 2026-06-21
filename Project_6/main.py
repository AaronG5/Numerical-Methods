# 1. Netiesinės ir tiesinės lygtys
#     2. Netiesinių lygčių sistema

import numpy as np
from scipy.optimize import fsolve
from simple_iteration import simple_iteration, easy_system, hard_system
from newton import newton_method, jacobian_easy, jacobian_hard

def easy_system_original(X):
   x = X[0]
   y = X[1]
   f1_res = x - 0.125 * (8 * x - 4 * x**2 + y**2 + 1)
   f2_res = y - 0.25 * (2 * x - x**2 + 4 * y - y**2 + 3)
   return (f1_res, f2_res)

def hard_system_original(X):
   x = X[0]
   y = X[1]
   z = X[2]
   f1_res = x**2 + y**2 + z**2
   f2_res = x * y + x * z - z - 0.5 * y
   f3_res = x**2 * y * z - y**2 - 3 * z**3 * y - (1/3)
   return (f1_res, f2_res, f3_res)

def main():
   X_easy = np.array((3.26, 5.27))
   X_hard = np.array((1.0, 1.0, 1.0))

   # Simple iteration method
   res_simple_iter_easy, iter_count_simple = simple_iteration(easy_system, X_easy)
   print(f'Simple iteration easy result: {res_simple_iter_easy}\nPlugging values into system: {np.round(easy_system_original(res_simple_iter_easy), 5)}\nIterations: {iter_count_simple + 1}\n')

   # Newton method
   res_newton_easy, iter_count_newton = newton_method(easy_system_original, jacobian_easy, X_easy)
   print(f'Newton\'s easy result: {res_newton_easy}\nPlugging values into system: {np.round(easy_system_original(res_newton_easy), 5)}\n')
   # res_newton_hard = newton_method(hard_system_original, jacobian_hard, X_hard)
   # print(f'Newton\'s hard result: {res_newton_hard}\nPlugging values into system: {np.round(hard_system_original(res_newton_hard), 5)}\n')
   
   # Separation method

   # Scipy
   res_scipy_easy = fsolve(easy_system_original, X_easy)
   print(f'Fsolve easy result: {res_scipy_easy}\nPlugging values into system: {np.round(easy_system_original(res_scipy_easy), 5)}\n')
   # res_scipy_hard = fsolve(hard_system_original, X_hard)
   # print(f'Fsolve hard result: {res_scipy_hard}\nPlugging values into system: {hard_system_original(res_scipy_hard)}')


if __name__ == '__main__':
   main()