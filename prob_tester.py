import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.committor_nn import NeuralNet
#from diffusion_map_mueller_nn import NeuralNet
from matplotlib import colors
from scipy import interpolate

input_size = 2
hidden_size = 50
output_size = 1
num_classes = 1
learning_rate = 0.000001


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load('/Users/luke/PycharmProjects/MarylandREU/data/net_mueller_b=0.1_art_temp=0.05_n=1000000_step=5_hs=50_layers=2')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

file = open('/Users/luke/PycharmProjects/MarylandREU/data/mueller_metadynamics_b=0.1_n=200000_precomputed.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
rows2 = []
cutoff = False
for row in csvreader:
    if row[0] == "S":
        cutoff = True
    elif not cutoff:
        rows.append(row)
    else:
        rows2.append(row)
file.close()

samples = np.zeros((len(rows), len(rows[0])))
for i in range(len(rows)):
    for j in range(len(rows[i])):
        samples[i][j] = float(rows[i][j])

n = 200000
crit = 0

model_vals = model(torch.tensor(samples,dtype=torch.float)).detach().numpy()
for s in range(len(samples)):
    if np.abs(model_vals[s]) < 0.1:
        crit+=1
print(crit/n)