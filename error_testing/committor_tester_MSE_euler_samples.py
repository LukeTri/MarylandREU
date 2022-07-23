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
hidden_size = 10
output_size = 1
num_classes = 1
learning_rate = 0.000001


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load('/Users/luke/PycharmProjects/MarylandREU/data/net_mueller_b=0.1_metadynamics_n=200000_step=1_hs=50_layers=2')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

file = open('/data/samples/mueller_metadynamics_b=0.033_n=200000_precomputed.csv')
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


file = open('/data/fe_results/fe_mueller_b=0.1.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
        rows.append(row)
file.close()

fem_vals = np.zeros(len(rows))

fem_samples = np.zeros((len(rows), 2))

for i in tqdm(range(len(rows))):
    fem_samples[i] = np.array([rows[i][0], rows[i][1]], dtype=float)
    fem_vals[i] = float(rows[i][2])

plt.scatter(fem_samples[:,0], fem_samples[:,1],c = fem_vals)
plt.show()

interp_vals = interpolate.griddata(fem_samples, fem_vals, samples)
ind = np.invert(np.isnan(interp_vals))
interp_vals = interp_vals[ind]

MSE = 0

model_vals = model(torch.tensor(samples, dtype=torch.float)).detach().numpy()
model_vals = model_vals[ind]

ind = mp.MuellerPotentialVectorized(samples[ind]) < -36

model_vals = model_vals[ind]
interp_vals = interp_vals[ind]

x = (interp_vals - model_vals) ** 2

print(min(x))
print(max(x))

RMSE = np.sqrt((np.sum((interp_vals - model_vals) ** 2 / len(interp_vals))))

print(RMSE)