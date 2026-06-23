import numpy as np
import pandas as pd
import os

def easy_system(X):
   x = X[0]
   y = X[1]
   x_res = 0.125 * (8 * x - 4 * x**2 + y**2 + 1)
   y_res = 0.25 * (2 * x - x**2 + 4 * y - y**2 + 3)
   return np.array((x_res, y_res))

def my_hard_system(X):
   x = X[0]
   y = X[1]
   z = X[2]
   x_res = -np.sqrt(-y**2 - z**2 + 3)
   y_res = (z * (x - 1)) / (x - 0.5)
   z_res = (np.sqrt(y * (x**4 * y + 12 * y**2 - 4)) + x**2 * y) / (6 * y)
   return np.array((x_res, y_res, z_res))

def save_steps(step_history, length):
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   os.makedirs(result_dir, exist_ok=True)
   filepath = os.path.join(result_dir, f'Simple_iteration_{length}.csv')
   df = pd.DataFrame(step_history)
   df.to_csv(filepath)

def simple_iteration(system, X, threshold = 1e-6, max_iter = 100):
   step_history = [X.copy()]
   for i in range(max_iter):
      X_old = X.copy()
      X = system(X_old)
      step_history.append(X.copy())
      if np.linalg.norm(X - X_old, np.inf) < threshold or not np.all(np.isfinite(X)):
         break 

   save_steps(step_history, len(X))

   steps = np.array(step_history)
   diffs = np.diff(steps, axis=0)
   norms = np.linalg.norm(diffs, axis=1, ord=2)

   return X, i, norms
