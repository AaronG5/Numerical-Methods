import numpy as np

def easy_system(X):
   x = X[0]
   y = X[1]
   x_res = 0.125 * (8 * x - 4 * x**2 + y**2 + 1)
   y_res = 0.25 * (2 * x - x**2 + 4 * y - y**2 + 3)
   return (x_res, y_res)

def hard_system(X):
   x = X[0]
   y = X[1]
   z = X[2]

def simple_iteration(system, X, max_iter = 1000):
   for i in range(max_iter):
      X = system(X)

   return X
