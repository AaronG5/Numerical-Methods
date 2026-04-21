# 6 (1, 5)
import numpy as np


def func(x):
   return (1 + x**2) / (7 + x**3)

def get_points(function, a, b): # Interval [a; b]
   x = np.linspace(a, b, 11)
   y = function(x)
   return x, y

class linear_interpolation():
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

def main():
   return 0

if __name__ == "__main__":
   main()