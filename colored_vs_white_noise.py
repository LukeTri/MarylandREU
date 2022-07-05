from Dataset import Dataset
import csv

from matplotlib.gridspec import GridSpec

import mueller_potential as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

n = 5
time_step = 10 ** -4
t = 5

omega = 1
epsilon = 1

res_white = np.zeros((n, int(t / time_step)))
res_colored = np.zeros((n, int(t / time_step)))

fig, ax = plt.subplots(2)

for i in range(n):
    for j in range(len(res_white[i])):
        if j == 0:
            res_white[i][j] = 0
        else:
            random_term = np.random.randn() * np.sqrt(time_step)
            res_white[i][j] = res_white[i][j-1] + random_term

for i in range(n):
    for j in range(len(res_colored[i])):
        if j == 0:
            res_colored[i][j] = 0
        else:
            random_term = np.random.randn() * np.sqrt(time_step)
            res_colored[i][j] = res_colored[i][j-1] - res_colored[i][j-1] * omega * time_step + random_term * np.sqrt(epsilon) * omega

time_steps = np.arange(0, t, time_step)


for i in range(n):
    ax[0].plot(time_steps, res_white[i], '-o',linewidth=0.2,markersize=0.2)
    ax[1].plot(time_steps, res_colored[i], '-o',linewidth=0.2,markersize=0.2)

plt.xlabel("time")
ax[0].set_ylabel("y(t)")
ax[0].set_title("White Noise")
ax[1].set_ylabel("y(t)")
ax[1].set_title("Colored Noise")
plt.show()
