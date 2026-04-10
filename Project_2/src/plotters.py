import matplotlib.pyplot as plt

def plot_cholesky(df):
   # Scaling factors
   const1_1 = df['Choleskio laik.'].iloc[-1] / df['N'].iloc[-1]**3
   const1_2 = df['Choleskio laik.'].iloc[-1] / df['N'].iloc[-1]**2
   const2_1 = df['Lygčių spr. laik.'].iloc[-1] / df['N'].iloc[-1]**2
   const2_2 = df['Lygčių spr. laik.'].iloc[-1] / df['N'].iloc[-1]

   # Cholesky plot
   fig1, ax1 = plt.subplots(figsize=(7, 5))
   ax1.plot(df['N'], df['Choleskio laik.'], label='Išmatuotas laikas')
   ax1.plot(df['N'], df['N']**3 * const1_1, label='O(N³)', linestyle='--', color='red') 
   ax1.plot(df['N'], df['N']**2 * const1_2, label='O(N²)', linestyle='--', color='orange')
   ax1.set_xlabel('N')
   ax1.set_ylabel('Laikas (ms)')
   ax1.set_title('Choleskio dekompozicija')
   ax1.legend()
   ax1.grid(True)
   fig1.savefig('Project_2/Cholesky_decomp.png', dpi=300)

   # Triangular plot
   fig2, ax2 = plt.subplots(figsize=(7, 5))
   ax2.plot(df['N'], df['Lygčių spr. laik.'], label='Išmatuotas laikas')
   ax2.plot(df['N'], df['N']**2 * const2_1, label='O(N²)', linestyle='--', color='red')
   ax2.plot(df['N'], df['N'] * const2_2, label='O(N)', linestyle='--', color='orange') 
   ax2.set_xlabel('N')
   ax2.set_ylabel('Laikas (ms)')
   ax2.set_title('Trikampių lygčių sprendimas')
   ax2.legend()
   ax2.grid(True)
   fig2.savefig('Project_2/Cholesky_triangular.png', dpi=300)

   # plt.show()

def plot_steepest_descent(df):

   fig, ax = plt.subplots(figsize=(7, 5))
   ax.plot(df['N'], df['Didžiausio nuolydžio laik.'], label="Išmatuotas laikas")
   # ax.plot(df['N'], df['N'])
   # TODO: Finish