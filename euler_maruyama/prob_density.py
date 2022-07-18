import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.committor_nn import NeuralNet
#from diffusion_map_mueller_nn import NeuralNet
from matplotlib import colors

file = open('../data/mueller_standard_b=0.02_n=1000000.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
        rows.append(row)
file.close()

arr = np.zeros((len(rows), len(rows[0])))
for i in range(len(rows)):
    for j in range(len(rows[i])):
        arr[i][j] = float(rows[i][j])

Npts,d = np.shape(arr)
X = arr[0:Npts, 0]
Y = arr[0:Npts, 1]

plt.scatter(X, Y)
mp.plot_contours()
plt.show()

hist, x_edge, y_edge = np.histogram2d(X,Y, bins=50, normed=True)
X,Y = np.meshgrid(x_edge[:-1], y_edge[:-1])
plt.pcolormesh(X,Y,hist)

expected_densities = np.zeros((len(hist), len(hist[0])))
#for i in range(len(x_edge) - 1):
    #for j in range(len(y_edge) - 1):


plt.show()