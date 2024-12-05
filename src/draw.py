import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print('Usage: python draw.py <output_file> <train_data>')
    sys.exit(1)

data_file = sys.argv[1]

tData = pd.read_csv(data_file, delim_whitespace=True, header=None, names=['x', 'y', 'z'])

x1, y1 = np.meshgrid(np.arange(0, 1.001, 0.001), np.arange(0, 1.001, 0.001))

z1 = griddata((tData['x'], tData['y']), tData['z'], (x1, y1), method='linear')

plt.contour(x1, y1, z1, levels=[0], linewidths=1)

plt.scatter(tData['x'], tData['y'], c=tData['z'], cmap='viridis', edgecolor='k')

plt.axis('equal')

plt.xlabel('x')
plt.ylabel('y')

# 8. Add a title to the plot
plt.title('Contour plot with scatter')

# 9. Show the plot
plt.savefig(f'{data_file}.png')
