import numpy as np
import pandas as pd
import os

def easy_system(X):
   x = X[0]
   y = X[1]
   x_res = 0.125 * (8 * x - 4 * x**2 + y**2 + 1)
   y_res = 0.25 * (2 * x - x**2 + 4 * y - y**2 + 3)
   return np.array((x_res, y_res))

def hard_system(X):
   x = X[0]
   y = X[1]
   z = X[2]

def save_steps(step_history):
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   os.makedirs(result_dir, exist_ok=True)
   filepath = os.path.join(result_dir, 'Paprastosios_iteracijos.csv')
   df = pd.DataFrame(step_history)
   df.to_csv(filepath)

def simple_iteration(system, X, threshold = 1e-6, max_iter = 1000):
   step_history = [X]
   for i in range(max_iter):
      X_old = X.copy()
      X = system(X_old)
      step_history.append(X)
      if np.linalg.norm(X - X_old, np.inf) < threshold:
         break 

   save_steps(step_history)

   return X, i
