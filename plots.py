import numpy as np
import matplotlib.pyplot as plt

labels = ['Prototype A', 'Prototype B']
x_pos = np.arange(len(labels))
CTEs = [4, 5]
error = [2.94, 2.45]

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('Coefficient of Thermal Expansion (\degreeC−1)')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()