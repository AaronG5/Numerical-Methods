# 6 (1, 5)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import interp1d, CubicSpline, lagrange
import os

def func(x):
   return (1 + x**2) / (7 + x**3)

def func_deriv(x):
   return -(x**4 + 3 * x**2 - 14 * x) / (7 + x**3)**2

def div_diff(x, y): # Divided difference of n-th row
   if len(x) == 2:
      return (y[-1] - y[0]) / (x[-1] - x[0])
   
   return (div_diff(x[1:], y[1:]) - div_diff(x[:-1], y[:-1]))/ (x[-1] - x[0])

def get_points(function, a, b): # Intervalas [a; b]
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

   def plot(self, result_dir):
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

      plt.plot(self.x, self.y, 'o', color='red', label='Mazgai')
      
      interval = np.linspace(self.x[0], self.x[-1], 100)
      plt.plot(interval, func(interval), color='orange', alpha=0.8, label='Duota funkcija')
      plt.legend()
      
      graph_filepath = os.path.join(result_dir, 'Linear_interpolation.png')
      plt.savefig(graph_filepath, dpi=300)
      plt.show()

class QuadraticSpline():
   def __init__(self, x, y):
      self.x = x
      self.y = y
      self.coef = []

   def find_coefficients(self, e):
      e = np.tan(np.radians(e))
      self.coef = []
      for i in range(10):
         j = max(1, min(i, 9))
         A = np.matrix(((self.x[i]**2, self.x[i], 1), 
                        (self.x[i+1]**2, self.x[i+1], 1), 
                        (2 * self.x[j], 1, 0)))
         B = np.array((self.y[i], self.y[i+1], e))
         solution = np.linalg.solve(A, B)
         self.coef.append(solution) # Choleskio ir greičiausio nuolydžio metodai čia netinka
         e = 2 * self.coef[i][0] * self.x[i+1] + self.coef[i][1]

   def find_value(self, x):
      if x < self.x[0] or x > self.x[-1]:
         raise ValueError(f"x={x} is outside the interpolation interval")
      
      for i in range(10):
         if self.x[i] <= x <= self.x[i+1]:
            return self.coef[i][0] * x**2 + self.coef[i][1] * x + self.coef[i][2]
         
   def _get_spline_points(self):
      x_all, y_all = [], []
      for i in range(10):
         x_subint = np.linspace(self.x[i], self.x[i+1], 20)
         y_subint = self.coef[i][0] * x_subint**2 + self.coef[i][1] * x_subint + self.coef[i][2]
         x_all.append(x_subint)
         y_all.append(y_subint)
      return np.concatenate(x_all), np.concatenate(y_all)

   def plot(self, result_dir):
      plt.figure(figsize=(6, 4))
      plt.title("Kvadratinis interpoliavimas splainais")
      plt.xlim((self.x[0] - 0.1, self.x[-1] + 0.1))
      plt.ylim((min(self.y) - 0.01, max(self.y) + 0.01))

      x_sp, y_sp = self._get_spline_points()
      plt.plot(x_sp, y_sp, linestyle='solid', color='blue', label='Kvadratinis splainas')
      plt.plot(self.x, self.y, 'o', color='red', label='Mazgai')
      
      interval = np.linspace(self.x[0], self.x[-1], 100)
      plt.plot(interval, func(interval), color='orange', alpha=0.8, label='Duota funkcija')
      plt.legend()
      graph_filepath = os.path.join(result_dir, 'Quadratic_Splines.png')
      plt.savefig(graph_filepath, dpi=300)
      plt.show()

   def animate(self, result_dir, alpha_range=range(-89, 90, 1)):
      fig, ax = plt.subplots(figsize=(6, 4))
      interval = np.linspace(self.x[0], self.x[-1], 100)

      ax.plot(interval, func(interval), color='orange', alpha=0.8, label='Duota funkcija')
      ax.plot(self.x, self.y, 'o', color='red', label='Interpoliavimo mazgai')
      spline_line, = ax.plot([], [], color='blue', label='Kvadratinis splainas')
      alpha_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

      ax.set_xlim(self.x[0] - 0.1, self.x[-1] + 0.1)
      ax.set_ylim(min(self.y) - 0.01, max(self.y) + 0.01)
      ax.legend()

      def update(alpha):
         self.find_coefficients(alpha)
         x_sp, y_sp = self._get_spline_points()
         spline_line.set_data(x_sp, y_sp)
         alpha_text.set_text(f'α = {alpha}°')
         return spline_line, alpha_text

      ani = animation.FuncAnimation(fig, update, frames=alpha_range, blit=True)
      ani.save(os.path.join(result_dir, 'spline_animation.gif'), writer='pillow', fps=15)
      plt.show()

def interpolate_built_in(x, y, result_dir):
   x_plot = np.linspace(x[0], x[-1], 100)
   i = 1
   for method in (interp1d, CubicSpline, lagrange): # lagrange sukuria vieną (n-1-ojo laipsnio) polinomą
      plt.figure(figsize=(6, 4))
      plt.title(f'Python funckijos: {i} {method.__name__}')
      plt.xlim((x[0] - 0.1, x[-1] + 0.1))
      plt.ylim((min(y) - 0.01, max(y) + 0.01)) 

      if method == interp1d:
         m = method(x, y, kind='quadratic')
      else:
         m = method(x, y)

      plt.plot(x_plot, m(x_plot), label=method.__name__)
      plt.plot(x_plot, func(x_plot), color='orange', alpha=0.8, label='Duota funkcija')
      plt.plot(x, y, 'o', color='red', label='Mazgai')
      plt.legend()

      filename = f'{i}_{method.__name__}.png'
      graph_filepath = os.path.join(result_dir, filename)
      plt.savefig(graph_filepath, dpi=300)
      i += 1

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

   linear_interpolation.plot(result_dir)

   quad_spline = QuadraticSpline(x, y)
   quad_spline.find_coefficients(-9) # -9 laipsniai tinka labiausiai
   print(quad_spline.find_value(2.8))
   quad_spline.plot(result_dir)

   quad_spline.animate(result_dir)

   interpolate_built_in(x, y, result_dir)

   return 0

if __name__ == "__main__":
   main()
   