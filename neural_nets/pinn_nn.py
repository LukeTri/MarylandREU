from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from Dataset import Dataset

import torch.nn as nn
import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
import torch

n = 100
x_start = -1.5
x_end = 1.5
y_start = -0.5
y_end = 2

x = np.linspace(x_start, x_end, n)
y = np.linspace(y_start, y_end, n)

grid = np.meshgrid(x, y)

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data"
NN_PATH = "/net_mueller_b=0.1_art_temp=0.05_n=1000000_step=5_hs=50_layers=2"
EM_PATH = "/face_standard=0.33_n=1000000.csv"

MUELLERMINA = torch.tensor([0.62347076, 0.02807048])
MUELLERMINB = torch.tensor([-0.55821361, 1.44174872])

xa=-3
ya=3
xb=0
yb=4.5
FACEMINA = torch.tensor([xa, ya])
FACEMINB = torch.tensor([xb, yb])

potential_func = "face"
art_temp = False
metadyanamics = False


radius = 0.1
epsilon = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 2
hidden_size = 10
output_size = 1
num_epochs = 50
batch_size = 100000
learning_rate = 0.1
num_classes = 1
momentum = 0.90

# Sampling parameters
step_size = 100
n = 100000
x = np.array([0, 0])
omega = 5
sigma = 0.05
b = 1 / 10
b_prime = 1 / 20
h = 10 ** -5



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.tanh1 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, num_classes)

        self.sig3 = nn.Sigmoid()

    def forward(self, x):
        x.requires_grad = True
        self.fc1.weight.requires_grad = True
        self.fc3.weight.requires_grad = True

        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.sig3(out)
        out = out.squeeze()
        out = (1 - chi_A(x)) * ((1 - chi_B(x)) * out + chi_B(x))

        return out


def chi_A_s(x):
    m = torch.nn.Tanh()
    if potential_func == "face":
        return 0.5 - 0.5 * m(1000 * ((x - FACEMINA).pow(2).sum() - (radius + 0.02) ** 2))
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINA).pow(2).sum() - (radius + 0.02) ** 2))


def chi_B_s(x):
    m = torch.nn.Tanh()
    if potential_func == "face":
        return 0.5 - 0.5 * m(1000 * ((x - FACEMINB).pow(2).sum() - (radius + 0.02) ** 2))
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINB).pow(2).sum() - (radius + 0.02) ** 2))


def chi_A(x):
    m = torch.nn.Tanh()
    if potential_func == "face":
        return 0.5 - 0.5 * m(1000 * ((x - FACEMINA).pow(2).sum(1) - (radius + 0.02) ** 2))
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINA).pow(2).sum(1) - (radius + 0.02) ** 2))


def chi_B(x):
    m = torch.nn.Tanh()
    if potential_func == "face":
        return 0.5 - 0.5 * m(1000 * ((x - FACEMINB).pow(2).sum(1) - (radius + 0.02) ** 2))
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINB).pow(2).sum(1) - (radius + 0.02) ** 2))


def main():
    file = open(FILE_PATH + EM_PATH)
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

    arr = np.zeros((len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            arr[i][j] = float(rows[i][j])

    if len(rows2) == 0:
        updaters = []
    else:
        updaters = np.zeros((len(rows2), len(rows2[0])))
        for i in range(len(rows2)):
            for j in range(len(rows2[i])):
                updaters[i][j] = float(rows2[i][j])

    Npts,d = np.shape(arr)
    X = arr[0:Npts:5, 0]
    Y = arr[0:Npts:5, 1]
    print(np.shape(X))

    plt.scatter(X, Y)
    #for i in range(len(updaters)):
        #plt.plot(updaters[i][0], updaters[i][1], markersize=20, marker="o")
    if potential_func == "face":
        fp.plot_contours()
    else:
        mp.plot_contours()
    plt.show()

    x = np.linspace(-1,1,50)
    y = np.linspace(-1,1,50)
    x_grid, y_grid = np.meshgrid(x,y)
    v_mpot_grid = np.zeros((len(x_grid), len(x_grid[0])))
    for i in range(len(x_grid)):
        for j in range(len(x_grid[i])):
            temp = torch.tensor(np.array([x_grid[i][j], y_grid[i][j]]))
            temp = temp.unsqueeze(0)
            if potential_func == "face":
                v_mpot_grid[i][j] = 0
            else:
                v_mpot_grid[i][j] = mp.get_updated_offset_gaussian(temp, updaters)

    plt.contour(x_grid, y_grid, v_mpot_grid, np.linspace(np.amin(v_mpot_grid), np.amax(v_mpot_grid), 20))
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(training_generator)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(training_generator):
            optimizer.zero_grad()
            outputs = model(samples)
            outputs = torch.autograd.grad(outputs, samples, allow_unused=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
            outputs = outputs.pow(2).sum(1)
            if art_temp:
                if potential_func == "face":
                    outputs = outputs * torch.exp(-(b - b_prime) * torch.tensor(fp.face_non_vectorized(
                        samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy())))
                else:
                    outputs = outputs * torch.exp(-(b - b_prime) * torch.tensor(mp.MuellerPotentialNonVectorized(
                        samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy())))
            elif metadyanamics:
                if potential_func == "face":
                    pass
                else:
                    outputs = outputs * torch.exp(b * (mp.get_updated_offset_gaussian(samples, updaters)))

            loss = torch.sqrt(outputs.sum())
            loss.backward()

            optimizer.step()
            # Backward and optimize

            if (epoch + 1) % 5 == 0 and i % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item() / batch_size))
        # scheduler.step()
        # if (epoch + 1) % 5 == 0:
            # print(scheduler.state_dict()['_last_lr'])

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, FILE_PATH + NN_PATH)
    # data/mueller_model_standard_b=0.033_n=100000_hs=20_ep=20_lr=0.000001_bs=10000.pth

    if potential_func == "face":
        m = np.arange(-5, 4, 0.1)
        p = np.arange(-3, 7, 0.1)

    else:
        p = np.arange(-.5, 2, 0.05)
        m = np.arange(-1.5, 1.5, 0.05)

    X, Y = np.meshgrid(m, p)
    Z = np.zeros((len(p), len(m)))
    for i in range(len(m)):
        for j in range(len(p)):
            tens = torch.tensor([[X[j][i], Y[j][i]]])
            tens = tens.float()
            Z[j][i] = model(tens)

    plt.pcolormesh(X, Y, Z, shading='gouraud')
    plt.colorbar()
    if potential_func == "face":
        fp.plot_contours()
    else:
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