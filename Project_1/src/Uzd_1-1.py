# 6 variant

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

def func(x):
   return x + 0.25 - np.tan(x)

def gunc_1(x):
   return np.tan(x) - 0.25

def gunc_2(x):
   return np.arctan(x + 0.25)

def gunc_1_deriv(x):
   return 1 + np.tan(x)**2

def gunc_2_deriv(x):
   return 1 / (1 + (x + 0.25)**2)

def simple_iteration_method(x_range):
   iteration = 0
   iteration_table = []
   q = None
   gunc = None

   # Išrenkame kurį g(x) naudosime ir randame q
   max_deriv = np.max(gunc_1_deriv(x_range))
   if max_deriv < 1:
      q = max_deriv
      gunc = gunc_1
   else:
      max_deriv = np.max(gunc_2_deriv(x_range))
      if max_deriv < 1:
         q = max_deriv
         gunc = gunc_2

   x_0 = x_range[0]
   x_1 = None

   iteration_table.append([iteration, x_0, None])

   plt.xlim((0, 1.5))
   plt.ylim((-2, 2))
   plt.plot(x_range, func(x_range), color='purple', label='f(x)')
   plt.plot(x_range, x_range, color='green', label='y = x')
   plt.plot(x_range, gunc(x_range), color='blue', label='g(x)')

   # Randame sprendinio įvertį
   while abs(gunc(x_0) - x_0) > (1-q) / q * eps:
      x_1 = gunc(x_0)
      iteration += 1
      iteration_table.append([iteration, x_1, abs(x_1 - x_0)])
      plt.plot([x_0, x_0], [x_0, x_1], color='red', linestyle='--')
      plt.plot([x_0, x_1], [x_1, x_1], color='red', linestyle='--')
      plt.plot(x_0, x_1, '.', color='red')
      x_0 = x_1

   plt.plot(x_0, func(x_0), 'o', color='orange', label=f'Sprendinys {x_0:.5f}')
   
   return pd.DataFrame(iteration_table, columns=['Iter. i', 'x_i', '|x_i - x_i-1|'])

def secant_method(x_range):
   iteration = 1
   iteration_table = []

   # Pradiniai 2 taškai
   x_0 = 1.5
   x_1 = 1.4
   secant_points = []

   # Randame sprendinio įvertį
   # while abs(func(x_1)) > eps:
   while abs(x_1 - x_0) > eps:
      secant_points.append((x_0, x_1))
      iteration_table.append([iteration, x_0, x_1, func(x_1)])
      iteration += 1
      x_new = x_1 - func(x_1) * (x_1 - x_0) / (func(x_1) - func(x_0))
      x_0 = x_1 
      x_1 = x_new

   plt.xlim((0, 2))
   plt.ylim((-10, 6))
   plt.plot(x_range, func(x_range), color='purple', label='f(x)')

   # Nubraižome kirstines
   for x1, x2 in secant_points:
      m = (func(x2) - func(x1)) / (x2 - x1)
      b = func(x1) - m * x1
      secant_values = m * x_range + b
      plt.plot(x_range, secant_values)
      plt.plot([x2, x2], [func(x2), 0.0], color='red', linestyle='--')

   plt.plot(x_1, func(x_1), 'o', color='orange', label=f'Sprendinys {x_1:.5f}')

   return pd.DataFrame(iteration_table, columns=['Iter. i', 'x_i', 'x_i+1', 'f(x_i+1)'])

eps = 0.001

x = np.linspace(0, 1.5, 2000)
y = x + 0.25 - np.tan(x)

result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
os.makedirs(result_dir, exist_ok=True)

for method_name, method_func in [
   ('Paprastųjų Iteracijų Metodas', simple_iteration_method),
   ('Kirstinių Metodas', secant_method)
]:
   graph_filepath = os.path.join(result_dir, method_name.replace(' ', '_').replace('ų', 'u') + '.png')
   table_filepath = os.path.join(result_dir, method_name.replace(' ', '_').replace('ų', 'u') + '.csv')

   plt.figure(figsize=(6, 4))
   plt.title(method_name)

   iteration_table = method_func(x)
   iteration_table.to_csv(table_filepath, index=False, na_rep='')
   
   plt.legend()
   plt.savefig(graph_filepath, dpi=300)
   plt.show()
   plt.close()
