# 6. f(x) = x * e**(-x**2)
# 2. Desiniuju staciakampiou metodas
# Intervalas: [1, 3]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def func(x):
   return x * np.e**(-x**2)

def deriv(x):
   return np.e**(-x**2) * (1 - 2 * x**2)

def find_M(a, b):
   x = np.arange(a, b+1)
   return np.max(np.abs(deriv(x)))

def integrate(a, b, n, M):
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

def main():
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'res')
   os.makedirs(result_dir, exist_ok=True)
   real_area = 0.1838780157

   a = 1
   b = 3
   M = find_M(a, b)

   n_arr = np.arange(20, 201, 20)

   rows = []

   for n in n_arr:
      est_area, est_error = integrate(a, b, n, M)
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

if __name__ == "__main__":
   main()