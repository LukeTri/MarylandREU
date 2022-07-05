import math
import time

from torch.autograd import Variable

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

MUELLERMINA = torch.tensor([0.62347076, 0.02807048])
MUELLERMINB = torch.tensor([-0.55821361, 1.44174872])
radius = 0.1
epsilon = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 2
hidden_size = 10
output_size = 1
num_epochs = 50
batch_size = 10000
learning_rate = 0.000001
num_classes = 1
momentum = 0.90

# Sampling parameters
step_size = 100
n = 100000
updaters = np.array([])
x = np.array([0, 0])
omega = 5
sigma = 0.05
b = 1 / 30
h = 10 ** -5


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.xavier_uniform(self.fc1.weight, gain=nn.init.calculate_gain('linear'))
        self.sig1 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform(self.fc3.weight, gain=nn.init.calculate_gain('linear'))
        self.sig3 = nn.Sigmoid()

    def get_committor(self, x):
        out = self.fc1(x)
        out = self.sig1(out)
        out = self.fc3(out)
        out = self.sig3(out)
        out = (1 - chi_A_s(x)) * ((1 - chi_B_s(x)) * out + chi_B_s(x))
        return out

    def forward(self, x):
        x.requires_grad = True
        self.fc1.weight.requires_grad = True
        self.fc3.weight.requires_grad = True

        out = self.fc1(x)
        out = self.sig1(out)
        out = self.fc3(out)
        out = self.sig3(out)
        out = out.squeeze()
        out = (1 - chi_A(x)) * ((1 - chi_B(x)) * out + chi_B(x))

        return out


def chi_A_s(x):
    m = torch.nn.Tanh()
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINA).pow(2).sum() - (radius + 0.02) ** 2))


def chi_B_s(x):
    m = torch.nn.Tanh()
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINB).pow(2).sum() - (radius + 0.02) ** 2))


def chi_A(x):
    m = torch.nn.Tanh()
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINA).pow(2).sum(1) - (radius + 0.02) ** 2))


def chi_B(x):
    m = torch.nn.Tanh()
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINB).pow(2).sum(1) - (radius + 0.02) ** 2))


def main():
    file = open('data/mueller_standard_b=0.033_n=500000.csv')
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


    X = arr[:, 0]
    Y = arr[:, 1]

    plt.scatter(X, Y)
    mp.plot_contours()
    plt.show()



    newX = np.zeros(len(X) // step_size)
    newY = np.zeros(len(Y) // step_size)
    for i in range(len(newX)):
        newX[i] = X[i * step_size]
        newY[i] = Y[i * step_size]

    s = np.vstack((X, Y)).T

    l = np.zeros(len(s))

    s = s.astype(np.float32)
    l = l.astype(np.float32)

    test_split = len(s) // 10
    x_valid = s[:test_split]
    y_valid = l[:test_split]

    x_train = s[test_split:]

    y_train = l[test_split:]

    training_set = Dataset(x_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(training_generator)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(training_generator):
            optimizer.zero_grad()
            outputs = model(samples)
            outputs = torch.autograd.grad(outputs, samples, allow_unused=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
            outputs = outputs.pow(2).sum(1)
            loss = outputs.sum()
            loss.backward()
            # Backward and optimize
            optimizer.step()
            if (epoch + 1) % 5 == 0 and i % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item() / batch_size))

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 'data/test')
    # data/mueller_model_standard_b=0.033_n=100000_hs=20_ep=20_lr=0.000001_bs=10000.pth

    m = np.arange(-1.5, 1.5, 0.05)
    p = np.arange(-.5, 2, 0.05)

    X, Y = np.meshgrid(m, p)
    Z = np.zeros((len(p), len(m)))
    for i in range(len(m)):
        for j in range(len(p)):
            tens = torch.tensor([X[j][i], Y[j][i]])
            tens = tens.float()
            Z[j][i] = model.get_committor(tens)

    plt.pcolormesh(X, Y, Z, shading='gouraud')
    plt.colorbar()
    mp.plot_contours()
    plt.ylabel("Y")
    plt.xlabel("X")


    plt.show()

# X = np.zeros(n)
# Y = np.zeros(n)
# for i in tqdm(range(n)):
#     x = mp.getNextIteration(x, h, updaters, sigma=sigma, omega=omega, b=b)
#     X[i] = x[0]
#     Y[i] = x[1]
# plt.scatter(X, Y)


if __name__ == "__main__":
    main()
