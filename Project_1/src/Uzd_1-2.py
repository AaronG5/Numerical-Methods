from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import numpy as np
import os

def func(x):
   return (x[0]**2 + x[1]**2 - 20), (x[0]**2 * x[1] + x[0] * x[1]**2 - 8)

result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')

x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)

roots = []
for x0 in x:
   for y0 in y:
      solution = fsolve(func, [x0, y0])
      solution = np.round(solution, 4)

      if not any(np.allclose(solution, r) for r in roots) \
      and np.allclose(func(solution), (0, 0), atol=0.001):
         roots.append(solution)

print(roots)

X, Y = np.meshgrid(x, y)
Z1, Z2 = func([X, Y])

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-50, 50)
ax.plot_surface(X, Y, Z1, alpha=0.5)
ax.plot_surface(X, Y, Z2, alpha=0.5)

for r in roots:
   ax.scatter3D(r[0], r[1], 0, color='black', s=60, alpha=1)

os.makedirs(result_dir, exist_ok=True)
plt.savefig(os.path.join(result_dir, 'Part_2.png'), dpi=300)
plt.show()