import numpy as np
import pandas as pd
import os

def save_steps(step_history, length):
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   os.makedirs(result_dir, exist_ok=True)
   filepath = os.path.join(result_dir, f'Separation_{length}.csv')
   df = pd.DataFrame(step_history)
   df.to_csv(filepath)

def newton(X, func, jacobian, i, threshold=1e-6, max_iter=100):
   for _ in range(max_iter):
      X_old = X[i]
      d = jacobian(X)[i][i]
      if abs(d) < 1e-10:
         raise ZeroDivisionError

      X[i] = X[i] - (func(X)[i] / d)
      
      if not np.isfinite(X[i]):
         raise FloatingPointError
      
      if abs(X[i] - X_old) < threshold:
         break

   return X[i]

def separation_method(system, jacobian, X, threshold=1e-6, max_iter=100):
   length = len(X)
   step_history = [X.copy()]

   for i in range(max_iter):
      X_old = X.copy()
      for j in range(length):
         X[j] = newton(X, system, jacobian, j)
      step_history.append(X.copy())
      if np.linalg.norm(X - X_old, 2) < threshold:
         break

   save_steps(step_history, length)

   steps = np.array(step_history)
   diffs = np.diff(steps, axis=0)
   norms = np.linalg.norm(diffs, axis=1, ord=2)

   return X, i, norms
