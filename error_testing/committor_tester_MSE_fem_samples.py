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
from scipy import interpolate

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data"
NN_PATH = "/net_pinn_uniform"
FE_PATH = "/fe_mueller_b=0.1.csv"

input_size = 2
hidden_size = 10
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


file = open(FILE_PATH + FE_PATH)
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

MSE = 0

model_vals = model(torch.tensor(fem_samples, dtype=torch.float)).detach().numpy()

ind = fp.face_vectorized(fem_samples) < 5

model_vals = model_vals[ind]
fem_vals = fem_vals[ind]

x = (fem_vals - model_vals) ** 2

print(np.mean(x))
print(np.sum(x))

RMSE = np.sqrt(np.sum((fem_vals - model_vals) ** 2 / len(fem_vals)))

print(RMSE)