# 6 (1, 5)
import matplotlib.pyplot as plt
import numpy as np
import os

def func(x):
   return (1 + x**2) / (7 + x**3)

def div_diff(x, y): # Divided difference of n-th row
   if len(x) == 2:
      return (y[-1] - y[0]) / (x[-1] - x[0])
   
   return (div_diff(x[1:], y[1:]) - div_diff(x[:-1], y[:-1]))/ (x[-1] - x[0])

def get_points(function, a, b): # Interval [a; b]
   x = np.linspace(a, b, 11)
   y = function(x)
   return x, y

class LinearInterpolation():
   def __init__(self, x, y):
      self.x = x
      self.y = y
      self.m = self.create_slope_coefficients(x, y) # Tiesių nuolydžio koeficientai

   def create_slope_coefficients(self, x, y):
      return np.diff(y) / np.diff(x)
   
   def evaluate(self, x):
      L = None
      
      if x < self.x[0] or x > self.x[-1]:
         raise ValueError(f"x={x} is outside the interpolation interval")
      
      for i in range(10):
         if self.x[i] <= x <= self.x[i+1]:
            L = self.y[i] + self.m[i] * (x - self.x[i])

            j = max(1, min(i, 9))
            M_2 = 2 * abs(div_diff(self.x[j-1:j+2], self.y[j-1:j+2]))
            estimated_error = 0.125 * (self.x[j+1] - self.x[j])**2 * M_2
            real_error = abs(L - func(x))
            return L, estimated_error, real_error

   def plot(self, result_dir, a, b):
      plt.figure(figsize=(6, 4))
      plt.title("Tiesinio interpoliavimo polinomas")
      plt.xlim((self.x[0] - 0.1, self.x[-1] + 0.1))
      plt.ylim((min(self.y) - 0.01, max(self.y) + 0.01))
      
      for i in range(10):
         plt.plot(
            (self.x[i], self.x[i+1]), 
            (func(self.x[i]), func(self.x[i+1])), 
            linestyle='solid', 
            color='blue', 
            label='Dalimis tiesinis interpoliavimas' if i == 0 else '_nolegend_'
         )

      plt.plot(self.x, self.y, 'o', color='red', label='Interpoliavimo mazgai')
      
      interval = np.linspace(a, b, 100)
      plt.plot(interval, func(interval), color='orange', alpha=0.8, label='Duota funkcija')
      plt.legend()
      
      graph_filepath = os.path.join(result_dir, 'Linear_interpolation.png')
      plt.savefig(graph_filepath, dpi=300)
      plt.show()

def main():
   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
   os.makedirs(result_dir, exist_ok=True)

   a = -1
   b = 4

   x, y = get_points(func, a, b)
   linear_interpolation = LinearInterpolation(x, y)

   for point in (2.8, 1):
      result, est_error, real_error = linear_interpolation.evaluate(point)
      print(f'Test point:      {point}\n'
            f'Result:          {result:.5f}\n' \
            f'Estimated error: {est_error:.5f}\n' \
            f'Actual error:    {real_error:.5f}\n' \
            f'Difference:      {abs(est_error - real_error):.5f}\n')

   linear_interpolation.plot(result_dir, a, b)

   return 0

if __name__ == "__main__":
   main()
   