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

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data"
OUTPUT_PATH = "/samples/mueller_iterative_n=600_bound=-15"
NN_PATH = "/nets/net_mueller_b=0.1_art_temp=0.05_n=1000000_step=5_hs=50_layers=2"

input_size = 2
hidden_size = 50
output_size = 1
num_classes = 1
learning_rate = 0.000001


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load(FILE_PATH + NN_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

n = 600
potential_func = "mueller"
adjustment_power = 1
base_prob = 0.4

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

ind = np.zeros(len(x),dtype=bool)
rands = np.random.rand(len(x))
for i in tqdm(range(len(x))):
    tens = torch.tensor([x[i], y[i]])
    tens = tens.float()
    tens = tens.unsqueeze(0)
    val = model(tens)
    prob = 1 if min(val, 1-val) > 0.1 else 0.2
    ind[i] = (prob > rands[i])

x, y = x[ind], y[ind]

data = np.vstack((x,y)).T
data = np.ndarray.tolist(data)

header = ['X','Y']
with open(FILE_PATH + OUTPUT_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
