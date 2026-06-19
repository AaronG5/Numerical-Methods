# 1. Netiesinės ir tiesinės lygtys
#     2. Netiesinių lygčių sistema

import numpy as np
from scipy.optimize import fsolve
from simple_iteration import simple_iteration, easy_system, hard_system

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
   X = (3.0, 3.0)
   res_1 = simple_iteration(easy_system, X)

   res_scipy = fsolve(easy_system_original, X)
   print(f'Simple iteration result: {res_1}\nPlugging values into system: {easy_system_original(res_1)}')
   print(f'Fsolve result: {res_scipy}\nPlugging values into system: {easy_system_original(res_scipy)}')


if __name__ == '__main__':
   main()