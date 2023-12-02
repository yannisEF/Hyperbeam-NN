import pickle
import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

parser = argparse.ArgumentParser()
parser.add_argument('--inputFolder', default='Results/', type=str)
parser.add_argument('--name', default='test_1701445754', type=str)
args = parser.parse_args()

list_to_load = list(range(25))

list_layers = []
for filename in list_to_load:
    folder = args.inputFolder + '/' + args.name + '/'
    with open(folder + str(filename) + '.pkl', 'rb') as handle:
        list_layers.append(pickle.load(handle))

v_min, v_max = np.min(list_layers), np.max(list_layers)

# 2D plot
fig = plt.figure(figsize=(8,8))
im=plt.imshow(list_layers[0], interpolation='none', vmin=v_min, vmax=v_max)

axidx1 = plt.axes([0.15, 0.05, 0.70, 0.04])
slidx1 = Slider(axidx1, 'index', 0, len(list_layers)-1, valinit=0, valfmt='%d', valstep=1)

def update2D(val):
    index = int(slidx1.val)
    im.set_array(list_layers[index])
slidx1.on_changed(update2D)

# 3D plot
list_layers3D = [layer[::-1] for layer in list_layers]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

X = list(range(len(list_layers3D[0][0])))
Y = list(range(len(list_layers3D[0])))
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, np.array(list_layers3D[0]))

axidx2 = plt.axes([0.15, 0.05, 0.70, 0.04])
slidx2 = Slider(axidx2, 'index', 0, len(list_layers3D)-1, valinit=0, valfmt='%d', valstep=1)
def update3D(val):
    index = int(slidx2.val)
    ax.clear()
    ax.plot_surface(X, Y, np.array(list_layers3D[index]))
    
    
slidx2.on_changed(update3D)

plt.show()