# 6 (1, 5)
import numpy as np


def func(x):
   return (1 + x**2) / (7 + x**3)

def get_points(function, a, b): # Interval [a; b]
   x = np.linspace(a, b, 11)
   y = function(x)
   return x, y

