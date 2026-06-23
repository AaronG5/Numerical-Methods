import matplotlib.pyplot as plt
import numpy as np
import os

def plot_diffs(simple_norms, newton, separation):
   plt.figure(figsize=(6, 4))
   plt.title('Metodų konvergavimas')
   plt.xlabel('Iteracija')
   plt.ylabel('2 laipsnio norma')

   plt.semilogy(range(len(simple_norms)), simple_norms, label='Paprastųjų iteracijų')
   plt.semilogy(range(len(newton)),       newton,       label='Niutono')
   plt.semilogy(range(len(separation)),   separation,   label='Išskaidymo')

   plt.legend()
   plt.grid(True, which='both', alpha=0.3)
   plt.tight_layout()

   result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
   os.makedirs(result_dir, exist_ok=True)
   filepath = os.path.join(result_dir, f'Convergence.pdf')

   plt.savefig(filepath)

   plt.show()
