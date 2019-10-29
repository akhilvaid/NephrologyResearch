#!/bin/python
# Requirements: A dataframe with 3 columns + 1 'CLUSTER' column
# In the absence of a 3rd column, the z value will be a constant

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d

outer_cmap = plt.get_cmap('tab20c')
outer_colors = outer_cmap([1, 5, 13])
palette = outer_colors.tolist()
cmap=ListedColormap(palette)

df = pd.read_csv('3DPCA.csv')
df = df.set_index('HADM_ID')
df.columns = ['A', 'B', 'C', 'CLUSTERS']

fig = plt.figure(figsize=(16, 16))
ax = plt.axes(projection="3d")

# Get rid of the ticks
ax.grid(True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Plot
ax.scatter3D(df['A'], df['B'], df['C'], c=df['CLUSTERS'], cmap=cmap)

# Legend
patches = []
for i in range(3):
    this_color = palette[i]
    label = 'CLUSTER ' + str(i + 1)
    this_patch = mpatches.Patch(color=this_color, label=label)
    patches.append(this_patch)
plt.legend(handles=patches)
plt.title('3D PROJECTION WITH SPECTRAL CLUSTERING')
