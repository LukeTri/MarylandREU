import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.diffusion_map_mueller_nn import NeuralNet as nn_dif
from matplotlib import colors

from neural_nets.net_architectures.layers2_act1ReLU_act2tanh import NeuralNet as net1
from neural_nets.net_architectures.layers2_act1tanh_act2tanh import NeuralNet as net2

nn_type = "1"

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data"
NN_PATH = "/nets/direct_sampling_4"
FE_PATH = "/fe_results/fe_mueller_b=0.1.csv"

input_size = 2
hidden_size = 20
output_size = 1
num_classes = 1
learning_rate = 0.1

potential_func = "mueller"


if nn_type == "1":
    model = net1(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
else:
    model = net2(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load(FILE_PATH + NN_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()


if potential_func == "face":
    m = np.arange(-5, 4, 0.1)
    p = np.arange(-3, 7, 0.1)

else:
    p = np.arange(-.5, 2, 0.05)
    m = np.arange(-1.5, 1.5, 0.05)

X, Y = np.meshgrid(m, p)
Z = np.zeros((len(p), len(m)))
for i in range(len(p)):
    for j in range(len(m)):
        angle = 0
        tens = torch.tensor([np.cos(angle)*X[i][j] + np.sin(angle)*Y[i][j], -np.sin(angle)*X[i][j] + np.cos(angle)*Y[i][j]])
        tens = tens.float()
        tens = tens.unsqueeze(0)
        Z[i][j] = model(tens)

plt.pcolormesh(X, Y, Z, shading='gouraud')
plt.colorbar()

if potential_func == "face":
    fp.plot_contours()
else:
    mp.plot_contours()

plt.title("Estimated Face Committor")

plt.show()

file = open(FILE_PATH + FE_PATH)
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

est_vals = np.zeros(len(rows))
act_vals = np.zeros(len(rows))


arr = torch.zeros((len(rows), len(rows[0])))

for i in tqdm(range(len(rows))):
    for j in range(len(rows[i])):
        arr[i][j] = float(rows[i][j])
    est_vals[i] = model(torch.tensor([[arr[i][0], arr[i][1]]], dtype=torch.float))
    act_vals[i] = arr[i][2]

X = arr[:, 0]
Y = arr[:, 1]

if potential_func == "face":
    ind = fp.face_non_vectorized(X,Y) < 5
else:
    ind = mp.MuellerPotentialNonVectorized(X,Y) < -36

est_vals = est_vals[ind]
act_vals = act_vals[ind]
X = X[ind]
Y = Y[ind]

diffs = est_vals - act_vals

print(max(diffs))
print(min(diffs))
print(np.average(np.abs(diffs)))

divnorm=colors.TwoSlopeNorm(vmin=min(diffs), vcenter=0., vmax=max(diffs))
plt.scatter(X, Y, c=diffs, cmap='bwr',norm=divnorm)
plt.colorbar()

if potential_func == "face":
    fp.plot_contours(x_start=min(X), x_end=max(X), y_start=min(Y), y_end=max(Y))
else:
    mp.plot_contours(x_start=min(X), x_end=max(X), y_start=min(Y), y_end=max(Y))

plt.title("Committor Error Face")
plt.show()