# 6. 
# Lygtis: y' = xy + (x/y)
# Pradine salyga: y(0) = 1
# Intervalas: x ∈ [0, 2]
# Metodai: 3, 5

import matplotlib.pyplot as plt
import numpy as np
import os

def diff(x, y):
   return x * y + (x / y)

def func(x, c):
   return np.sqrt(np.exp(x**2 + c) - 1)

def C(x, y):
   return np.log(y**2 + 1) - x**2

def simp_iter(x, y, h):
   eps = 0.0001
   y_next = y

   while True:
      y_prev = y_next
      y_next = y + 0.5 * h * (diff(x, y) + diff(x + h, y_next))

      if abs(y_next - y_prev) < eps:
         return y_next

def third_order_runge_kutta_method(a, b, y, h):
   graph = []

   for x in np.arange(a, b+h, h):
      graph.append((x, y))
      k1 = diff(x, y)
      k2 = diff(x + 0.5 * h, y + 0.5 * h * k1)
      k3 = diff(x + h, y + h * (-k1 + 2 * k2))

      y = y + (h / 6) * (k1 + 4 * k2 + k3)

   return graph

def symmetric_euler_method(a, b, y, h):
   graph = []

   for x in np.arange(a, b+h, h):
      graph.append((x, y))
      
      # y_est = simp_iter(x, y, h)
      # y = y + 0.5 * h * (diff(x, y) + diff(x + h, y_est))

      y = simp_iter(x, y, h)

   return graph

def plot_graph(graph_points, c, method_name, result_dir):
   plt.figure(figsize=(6, 4))
   plt.title(method_name)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.xlim((-0.05, 2.05))

   x_values = np.array([x for x, _ in graph_points])
   y_values = np.array([y for _, y in graph_points])

   plt.plot(x_values, y_values, label=method_name, color='blue') # Method
   plt.plot(x_values, y_values, '.', color='red') # Method
   plt.plot(x_values, func(x_values, c), label='Tikroji funkcija', color='orange', alpha=0.5)

   filename = method_name.replace(' ', '_')
   filename = filename + '.png'
   graph_filepath = os.path.join(result_dir, filename)
   plt.legend()
   plt.savefig(graph_filepath, dpi=300)
   plt.show()

def main():
   result_dir = os.path.dirname(os.path.abspath(__file__))
   os.makedirs(result_dir, exist_ok=True)
   
   y = 1.0
   a = 0.0
   b = 2.0
   h = 0.1

   c = C(a, y)

   method_1 = 'Tripakopis Rungės-Kuto metodas'
   graph_1 = third_order_runge_kutta_method(a, b, y, h)
   plot_graph(graph_1, c, method_1, result_dir)

   method_2 = 'Simetrinis Eulerio metodas'
   graph_2 = symmetric_euler_method(a, b, y, h)
   plot_graph(graph_2, c, method_2, result_dir)

   return 0

if __name__ == '__main__':
   main()