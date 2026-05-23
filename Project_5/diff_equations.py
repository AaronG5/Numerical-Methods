# 6. 
# Lygtis: y' = xy + (x/y)
# Pradine salyga: y(0) = 1
# Intervalas: x ∈ [0, 2]
# Metodai: 3, 5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def diff(x, y):
   return x * y + (x / y)

def func(x, c):
   return np.sqrt(np.exp(x**2 + c) - 1)

def C(x, y):
   return np.log(y**2 + 1) - x**2

def simp_iter(x, y, h):
   eps = 0.000001
   y_next = y

   while True:
      y_prev = y_next
      y_next = y + 0.5 * h * (diff(x, y) + diff(x + h, y_next))

      if abs(y_next - y_prev) < eps:
         return y_next

def third_order_runge_kutta_method(x_values, y, h):
   y_values = []

   for x in x_values:
      y_values.append(y)
      k1 = diff(x, y)
      k2 = diff(x + 0.5 * h, y + 0.5 * h * k1)
      k3 = diff(x + h, y + h * (-k1 + 2 * k2))

      y = y + (h / 6) * (k1 + 4 * k2 + k3)

   return y_values

def symmetric_euler_method(x_values, y, h):
   y_values = []

   for x in x_values:
      y_values.append(y)
      
      # y_est = simp_iter(x, y, h)
      # y = y + 0.5 * h * (diff(x, y) + diff(x + h, y_est))

      y = simp_iter(x, y, h)

   return y_values

def plot_graph(x_values, y_values, real_y_values, h, method_name, result_dir):
   filename = method_name.replace(' ', '_')
   graph_filename = filename + '.png'
   graph_filepath = os.path.join(result_dir, graph_filename)
   table_filename = filename + f'_h_{h}' + '.csv'
   table_filepath = os.path.join(result_dir, table_filename)

   plt.figure(figsize=(6, 4))
   plt.title(method_name)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.xlim((-0.05, 2.05))

   plt.plot(x_values, y_values, label=method_name, color='blue') # Method
   plt.plot(x_values, y_values, '.', color='red') # Method dots
   plt.plot(x_values, real_y_values, label='Tikroji funkcija', color='orange', alpha=0.5)

   plt.legend()
   plt.savefig(graph_filepath, dpi=300)
   # plt.show()

   err = np.abs(real_y_values - y_values)
   df = pd.DataFrame({
      'x': np.round(x_values, 10),
      'y_est': y_values,
      'y_real': real_y_values,
      'error': err
   })
   df.to_csv(table_filepath)

   return np.max(err)

def main():
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   os.makedirs(result_dir, exist_ok=True)
   
   y = 1.0
   a = 0.0
   b = 2.0
   c = C(a, y)
   h_values = (0.01, 0.05, 0.1)

   error_norms = []

   for h in h_values:
      x_values = np.linspace(a, b, int((b - a) / h) + 1)
      
      real_y_values = np.array(func(x_values, c))

      method_1 = 'Tripakopis Rungės-Kuto metodas'
      y_values_1 = third_order_runge_kutta_method(x_values, y, h)
      runge_kutta_err_norm = plot_graph(x_values, y_values_1, real_y_values, h, method_1, result_dir)

      method_2 = 'Simetrinis Eulerio metodas'
      y_values_2 = symmetric_euler_method(x_values, y, h)
      symmetric_euler_norm = plot_graph(x_values, y_values_2, real_y_values, h, method_2, result_dir)
      error_norms.append({
         'h': h,
         'runge_kutta_norm': runge_kutta_err_norm,
         'symmetric_euler_norm': symmetric_euler_norm
      })
   print(error_norms[2]['runge_kutta_norm'] / error_norms[1]['runge_kutta_norm'])
   print(error_norms[2]['symmetric_euler_norm'] / error_norms[1]['symmetric_euler_norm'])

   df = pd.DataFrame(error_norms)
   
   norm_filepath = os.path.join(result_dir, 'Max_normos.csv')
   df.to_csv(norm_filepath)
   # print(error_norms)
   return 0

if __name__ == '__main__':
   main()
