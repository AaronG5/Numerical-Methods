# 6 variant

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def func(x):
   return x + 0.25 - np.tan(x)

def deriv(x):
   return - (np.tan(x))**2

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
   
   iteration_table = pd.DataFrame(iteration_table, columns=['Iter. i', 'x_i', '|x_i - x_i-1|'])
   print(iteration_table.to_string(index=False, na_rep='', justify='left'))

def half_slice_search(x_range):
   a = x_range[0]
   b = x_range[-1]

   intervals = []

   c = (a + b) / 2

   while np.abs(func(c)) > eps:

      intervals.append((a, b))

      if func(a) * func(c) < 0:
         b = c
      elif func(b) * func(c) < 0:
         a = c

      c = (a + b) / 2

   plt.xlim((0, 2))
   plt.ylim((-10, 4))
   plt.plot(x_range, func(x_range), color='purple', label='f(x)')
   plt.plot(c, func(c), 'o', color='orange', label=f'Sprendinys {c:.5f}')

   interval_height = 1
   for a, b in intervals:
      interval = np.linspace(a, b, 100)
      interval_heights = np.full(len(interval), interval_height)
      plt.plot(interval, interval_heights)
      interval_height += 0.2

def newton_method(x_range):
   x_i = 1.5
   tangent_points = []
   
   step = abs(x_i - x_range[0])

   while step > eps:
      tangent_points.append((x_i, func(x_i)))

      x_new = x_i - func(x_i) / deriv(x_i)
      step = abs(x_new - x_i)
      x_i = x_new

   plt.xlim((0, 2))
   plt.ylim((-10, 6))
   plt.plot(x_range, func(x_range), color='purple', label='f(x)')

   for x_tan, y_tan in tangent_points:
      tangent_func = deriv(x_tan) * (x_range - x_tan) + y_tan
      plt.plot(x_range, tangent_func)

   plt.plot(x_i, func(x_i), 'o', color='orange', label=f'Sprendinys {x_i:.5f}')

def secant_method(x_range):
   iteration = 1
   iteration_table = []

   # Pradiniai 2 taškai
   x_0 = 1.5
   x_1 = 1.4
   secant_points = []

   # Randame sprendinio įvertį
   while abs(func(x_1)) > eps:
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

   iteration_table = pd.DataFrame(iteration_table, columns=['Iter. i', 'x_i', 'x_i+1', 'f(x_i+1)'])
   print(iteration_table.to_string(index=False, justify='left'))

eps = 0.001

x = np.arange(0, 1.5, 0.01)

y = x + 0.25 - np.tan(x)

for method_name, method_func in [
   ('Paprastųjų Iteracijų Metodas', simple_iteration_method),
   # ('Pusiaukirtos Metodas', half_slice_search),
   # ('Niutono Metodas', newton_method),
   ('Kirstinių Metodas', secant_method)
]:
   plt.figure(figsize=(6, 4))
   filepath = 'Project_1/' + method_name.replace(' ', '_').replace('ų', 'u') + '.png'
   plt.title(method_name)

   method_func(x)

   plt.legend()
   # plt.savefig(filepath, dpi=300)
   plt.show()
   plt.close()
