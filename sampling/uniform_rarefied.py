import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.committor_nn import NeuralNet
#from diffusion_map_mueller_nn import NeuralNet
from matplotlib import colors

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data/samples/mueller_rarefied_n=500_bound=-15"

n = 500
potential_func = "mueller"

x = np.linspace(-1.5, 1.5, n)
y = np.linspace(-0.5, 2, n)

x, y = np.meshgrid(x,y)

x = x.ravel()
y = y.ravel()

if potential_func == "face":
    ind = fp.face_non_vectorized(x,y) < 5
else:
    ind = mp.MuellerPotentialNonVectorized(x,y) < -15

x, y = x[ind], y[ind]
data = np.vstack((x,y)).T
data = np.ndarray.tolist(data)

header = ['X','Y']
with open(FILE_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
