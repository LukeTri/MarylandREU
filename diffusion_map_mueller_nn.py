import math
import time
from Dataset import Dataset

from matplotlib.gridspec import GridSpec

import mueller_potential as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

plt.style.use('seaborn')

MUELLERMINA = np.array([0.62347076, 0.02807048])
MUELLERMINB = np.array([-0.55821361, 1.44174872])

updaters = []
fig, ax = plt.subplots()
sigma = 0.2


def diffusion_map(index, lam, r, eigvals):
    ret = np.zeros(len(eigvals))
    i = 0
    t = np.ceil(np.real(np.log(sigma) / (np.log(lam[eigvals[len(eigvals)-1]]) - np.log(lam[eigvals[0]]))))
    for eigenval_num in eigvals:
        ret[i] = lam[eigenval_num]**t * r[index][eigenval_num]
        i += 1
    return ret


def get_updated_offset(x, y, updaters, omega=5, sigma=0.05):
    offset = 0
    for q in range(len(updaters)):
        xn1 = updaters[q][0]
        xn2 = updaters[q][1]
        offset += omega * np.e ** (-((x - xn1)**2 + (y - xn2)**2)/sigma)
    return offset


def convert_q_to_colors(q):
    c = np.zeros((len(q),3))
    for i in range(len(q)):
        c[i][0] = q[i]
        c[i][1] = 1 - q[i]
        if q[i] >= 1:
            c[i][0] = 1
            c[i][1] = 0
        if q[i] <= 0:
            c[i][0] = 0
            c[i][1] = 1
    return c

def eig_stuff(X_i, Y):
    X = np.zeros((len(Y), 2))
    for i in range(len(X)):
        X[i][0] = X_i[i]
        X[i][1] = Y[i]
    dist = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            dist[i][j] = np.linalg.norm(X[i] - X[j])

    drowmin = np.zeros(len(dist))
    for i in range(len(dist)):
        m = 1000
        for j in range(len(dist)):
            if i != j and dist[i][j] < m:
                drowmin[i] = dist[i][j]
    epsilon = 0.12

    k = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            k[i][j] = np.exp(-(dist[i][j]) ** 2 / epsilon)

    q = np.zeros(len(X))
    P = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        q[i] = np.sum(k[i])
        P[i] = k[i] / q[i]

    L = (P - np.identity(len(P))) / epsilon

    a_bool = np.zeros(len(X), dtype=bool)
    b_bool = np.zeros(len(X), dtype=bool)
    c_bool = np.zeros(len(X), dtype=bool)
    for i in range(len(a_bool)):
        if np.linalg.norm(X[i] - MUELLERMINA) < 0.5:
            a_bool[i] = True
        if np.linalg.norm(X[i] - MUELLERMINB) < 0.25:
            b_bool[i] = True
        if not a_bool[i] and not b_bool[i]:
            c_bool[i] = True
    LC_row = L[c_bool, :]

    LCC = LC_row[:, c_bool]
    LCB = LC_row[:, b_bool]

    ones = np.ones(len(LCB[0]))
    b = -LCB @ ones

    q = np.linalg.solve(LCC, b)
    return q, a_bool, b_bool, c_bool


def mse(input, target):
    return torch.sum((input-target) ** 2)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sig2(out)
        return out


def neural_net(x, y, probs):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 2
    hidden_size = 500
    output_size = 1
    num_epochs = 5000
    batch_size = 500
    learning_rate = 0.01
    num_classes = 1

    x = np.vstack((x, y)).T

    p = np.random.permutation(len(x))

    x = x[p]
    probs = probs[p]

    x = x.astype(np.float32)
    probs = probs.astype(np.float32)

    test_split = len(x) // 10
    x_valid = x[:test_split]
    y_valid = probs[:test_split]

    x_train = x[test_split:]
    y_train = probs[test_split:]

    training_set = Dataset(x_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

    y_train = y_train.unsqueeze(1)
    y_valid = y_valid.unsqueeze(1)
    #y_train = y_train.type(torch.LongTensor)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(training_generator)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(training_generator):
            labels = labels.to(device)

            # Forward pass
            outputs = model(samples)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    print(outputs[:5])
    print(labels[:5])
    print(outputs.shape)
    print(labels.shape)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 'data/mueller_model_diffusion_b=0.033_n=100000_hs=500_ep=5000_lr=0.01_bs=500.pth')

    return model


def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=5, b = 1/30):
    updaters = []
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        x = mp.getNextIteration(x, h, updaters,sigma=sigma, omega=omega,b=b)
        X[i] = x[0]
        Y[i] = x[1]

    step_size = 100
    newX = np.zeros(len(X) // step_size)
    newY = np.zeros(len(Y) // step_size)
    for i in range(len(newX)):
        newX[i] = X[i * step_size]
        newY[i] = Y[i * step_size]

    mp.plot_contours()
    c_probs, a_bool, b_bool, c_bool = eig_stuff(newX, newY)

    newX_C = newX[c_bool]
    newY_C = newY[c_bool]

    newX_B = newX[b_bool]
    newY_B = newY[b_bool]

    newX_A = newX[a_bool]
    newY_A = newY[a_bool]

    c_colors = convert_q_to_colors(c_probs)
    b_probs = np.ones(len(newX_B))
    a_probs = np.zeros(len(newX_A))

    x_data = np.concatenate((newX_A, newX_B, newX_C))
    y_data = np.concatenate((newY_A, newY_B, newY_C))
    probs = np.concatenate((a_probs, b_probs, c_probs))

    model = neural_net(x_data, y_data, probs)
    colors = convert_q_to_colors(probs)

    plt.scatter(x_data, y_data,s=100, c=colors)
    plt.show()


    m = np.arange(-1.5, 1.5, 0.05)
    p = np.arange(-.5, 2, 0.05)

    X, Y = np.meshgrid(m, p)
    Z = np.zeros((len(p), len(m)))
    for i in range(len(m)):
        for j in range(len(p)):
            tens = torch.tensor([X[j][i], Y[j][i]])
            tens = tens.float()
            Z[j][i] = model(tens)

    plt.pcolormesh(X, Y, Z, shading='gouraud')
    mp.plot_contours()
    plt.show()

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    createGraph(np.array([0, 0]), 10 ** -5, 100000, 0, 0, omega=5)

plt.show()
