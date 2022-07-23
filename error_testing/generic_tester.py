import colorsys
from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.committor_nn import NeuralNet
import csv

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data"
NN_PATH = "/nets/net_pinn_iter_4_epochs=4000"
SAMPLE_PATH = "/samples/mueller_standard_b=0.033_n=1000000.csv"

MUELLERMINA = np.array([0.62347076, 0.02807048])
MUELLERMINB = np.array([-0.55821361, 1.44174872])

input_size = 2
hidden_size = 10
output_size = 1
num_classes = 1
learning_rate = 0.000001

delta = 0.1
h = 10 ** -4
trials = 1000
step_size = 500
b = 1/10

potential_func = "mueller"

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load(FILE_PATH + NN_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

file = open(FILE_PATH + SAMPLE_PATH)
csvreader = csv.reader(file)
next(csvreader)
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

samples = samples[::step_size]


def getFirstMinimum(x):
    res = np.zeros(len(x),dtype=int)
    for i in range(2000):
        x = mp.get_next_iteration_vectorized(x, h, b=b)
        dist_a = np.linalg.norm(x - MUELLERMINA, axis=1) < delta
        dist_b = np.linalg.norm(x - MUELLERMINB, axis=1) < delta
        ind = res == 0
        edit_a = np.all([ind, dist_a], axis=0)
        edit_b = np.all([ind, dist_b], axis=0)
        res[edit_a] = 1
        res[edit_b] = 2
    print(np.bincount(res))
    return res


def getProbability(x):
    vals = np.zeros(len(x))
    for i in tqdm(range(trials)):
        ind = np.array(getFirstMinimum(x) - 1, dtype=bool)
        vals[ind] += 1
    vals = vals/trials
    return vals


observed = getProbability(samples)

estimated = model(torch.tensor(samples, dtype=torch.float)).detach().numpy()

if potential_func == "face":
    ind = fp.face_vectorized(samples) < 5
else:
    ind = mp.MuellerPotentialVectorized(samples) < -36

observed = observed[ind]
estimated = estimated[ind]

x = (observed - estimated) ** 2

print(np.mean(x))
print(np.sum(x))

RMSE = np.sqrt(np.sum((observed - estimated) ** 2 / len(observed)))

print(RMSE)