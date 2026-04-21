# 6 (1, 5)
import matplotlib.pyplot as plt
import numpy as np
import os.path as op

def func(x):
   return (1 + x**2) / (7 + x**3)

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
      # m = []
      # for i in range(10):
      #    m.append((y[i+1] - y[i]) / (x[i+1] - x[i]))
      # return m

      return np.diff(y) / np.diff(x) # Vectorization baby!
   
   def evaluate(self, x):
      if x < self.x[0] or x > self.x[-1]:
         raise ValueError(f"x={x} is outside the interpolation interval")
      
      for i in range(10):
         if self.x[i] <= x <= self.x[i+1]:
            return self.y[i] + self.m[i] * (x - self.x[i])

   def plot(self):
      result_dir = op.join(op.dirname(op.abspath(__file__)), '..', 'result')

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

      plt.legend()
      plt.show()
      plt.savefig()

def main():
   x, y = get_points(func, -1, 4)
   linear_interpolation = LinearInterpolation(x, y)

   print(linear_interpolation.evaluate(2.3))
   linear_interpolation.plot()

   return 0

if __name__ == "__main__":
   main()