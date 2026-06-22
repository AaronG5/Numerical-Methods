import numpy as np
import pandas as pd
import os

def jacobian_easy(X):
   x = X[0]
   y = X[1]
   f1_x = x
   f1_y = -0.25 * y
   f2_x = 0.5 * (x - 1)
   f2_y = 0.5 * y
   return np.array(((f1_x, f1_y), (f2_x, f2_y)))

def jacobian_hard(X):
   x = X[0]
   y = X[1]
   z = X[2]
   f1_x = 2 * x
   f1_y = 2 * y
   f1_z = 2 * z
   f2_x = y + z
   f2_y = x - 0.5
   f2_z = x - 1
   f3_x = 2 * x * y * z
   f3_y = x**2 * z + 2 * y - 3 * z**3
   f3_z = x**2 * y - 9 * z**2 * y
   return np.array(((f1_x, f1_y, f1_z), (f2_x, f2_y, f2_z), (f3_x, f3_y, f3_z)))

def save_steps(step_history):
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   os.makedirs(result_dir, exist_ok=True)
   filepath = os.path.join(result_dir, 'Newton.csv')
   df = pd.DataFrame(step_history)
   df.to_csv(filepath)

def newton_method(system, jacobian, X, threshold = 1e-6, max_iter = 100):
   step_history = [X.copy()]
   for i in range(max_iter):
      X_old = X.copy()
      F = np.array(system(X))
      J = jacobian(X)
      R = J @ X - F
      X = np.linalg.solve(J, R)
      step_history.append(X.copy())
      if np.linalg.norm(X - X_old, 2) < threshold:
         break

   save_steps(step_history)

   return X, i