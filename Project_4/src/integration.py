# 6. f(x) = x * e**(-x**2)
# 2. Desiniuju staciakampiou metodas
# Intervalas: [1, 3]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

I_count = 0

def func(x):
   return x * np.e**(-x**2)

def deriv(x):
   return np.e**(-x**2) * (1 - 2 * x**2)

def find_M(a, b):
   x = np.linspace(a, b, 1000)
   return np.max(np.abs(deriv(x)))

def find_area(a, b):
   global I_count
   I_count += 1
   return abs(b - a) * func(b)

def integrate(a, b, n):
   M = find_M(a, b)
   interval = np.linspace(a, b, n+1)
   h = (b - a) / n

   est_area = np.sum(func(interval[1:])) * h

   est_error = 0.5 * M * (b - a) * h 

   return est_area, est_error

def plot(data, result_dir):
   plt.figure(figsize=(6, 4))
   plt.title("Integravimo paklaidos")
   
   plt.plot(data['N'], data['Apytiksle paklaida'], label='Apytikslė paklaida')
   plt.plot(data['N'], data['Tikroji paklaida'], label='Tikroji paklaida')

   graph_filepath = os.path.join(result_dir, 'Integravimas.png')
   plt.savefig(graph_filepath, dpi=300)
   plt.show()

def adaptive_integrate(a, b, eps, I_full=None):
   if I_full is None:
      I_full = find_area(a, b)
   mid = (a + b) / 2
   I_left = find_area(a, mid)
   I_right = find_area(mid, b)

   if abs(I_full - I_left - I_right) < eps:
      return I_left + I_right
   else:
      return (adaptive_integrate(a, mid, eps, I_left) + 
              adaptive_integrate(mid, b ,eps, I_right))

def run_integral(a, b, n_arr, real_area, result_dir):
   rows = []

   for n in n_arr:
      est_area, est_error = integrate(a, b, n)
      rows.append({
         'N': n, 
         'Apytiksle reiksme': est_area,
         'Apytiksle paklaida': est_error,
         'Tikroji paklaida': abs(est_area - real_area)
         })


   table_filepath = os.path.join(result_dir, 'Integravimas.csv')
   df = pd.DataFrame(rows)
   df.to_csv(table_filepath)

   plot(df, result_dir)

   return 0

def run_adaptive_integral(a, b, real_area):
   global I_count
   eps = 0.001

   est_area = adaptive_integrate(a, b, eps)

   print(f'Apytiksle reiksme: {est_area}')
   print(f'Paklaida: {abs(est_area-real_area)}')
   print(f'Funkciju skaiciavimo kiekis: {I_count}')

   return 0

def main():
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'res')
   os.makedirs(result_dir, exist_ok=True)
   real_area = 0.1838780157

   a = 1
   b = 3

   n_arr = np.arange(20, 301, 20)

   run_integral(a, b, n_arr, real_area, result_dir)
   run_adaptive_integral(a, b, real_area, result_dir)

   return 0

if __name__ == "__main__":
   main()