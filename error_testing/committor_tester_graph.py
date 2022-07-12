import csv

from euler_maruyama import euler_maruyama_white_noise as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.committor_nn import NeuralNet
#from diffusion_map_mueller_nn import NeuralNet
from matplotlib import colors

input_size = 2
hidden_size = 10
output_size = 1
num_classes = 1
learning_rate = 0.1



model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load('data/test')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()



m = np.arange(-1.5, 1.5, 0.05)
p = np.arange(-.5, 2, 0.05)

X, Y = np.meshgrid(m, p)
Z = np.zeros((len(p), len(m)))
for i in range(len(m)):
    for j in range(len(p)):
        tens = torch.tensor([X[j][i], Y[j][i]])
        tens = tens.float()
        tens = tens.unsqueeze(0)
        Z[j][i] = model(tens)

plt.pcolormesh(X, Y, Z, )
plt.colorbar()

plt.show()

file = open('data/fe_mueller_b=0.033.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
        rows.append(row)
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

print(est_vals)
print(act_vals)

diffs = est_vals - act_vals

divnorm=colors.TwoSlopeNorm(vmin=min(diffs), vcenter=0., vmax=max(diffs))
plt.scatter(X, Y, c=diffs, cmap='bwr',norm=divnorm)
plt.colorbar()

mp.plot_contours(x_start=min(X), x_end=max(X), y_start=min(Y), y_end=max(Y))

plt.show()