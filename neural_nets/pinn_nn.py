from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from Dataset import Dataset

import torch.nn as nn
import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
import torch

n = 200
x_start = -1.5
x_end = 1.5
y_start = -0.5
y_end = 2

FILE_PATH = "/Users/luke/PycharmProjects/MarylandREU/data"
NN_PATH = "/net_pinn_EM_2"
EM_PATH = "/mueller_standard_b=0.033_n=1000000.csv"

MUELLERMINA = torch.tensor([0.62347076, 0.02807048])
MUELLERMINB = torch.tensor([-0.55821361, 1.44174872])

xa=-3
ya=3
xb=0
yb=4.5
FACEMINA = torch.tensor([xa, ya])
FACEMINB = torch.tensor([xb, yb])

potential_func = "mueller"

radius = 0.1
epsilon = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 2
hidden_size = 10
output_size = 1
num_epochs = 200
batch_size = 10000
learning_rate = 0.2
num_classes = 1
momentum = 0.90

# Sampling parameters
step_size = 100
b = 1 / 10

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

    Npts, d = np.shape(arr)
    X = arr[0:Npts:25, 0]
    Y = arr[0:Npts:25, 1]

    if potential_func == "face":
        fp.plot_contours()
    else:
        mp.plot_contours()

    plt.scatter(X,Y,marker='o',s=1)
    plt.show()

    X = np.ravel(X)
    Y = np.ravel(Y)

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

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.997)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(training_generator)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(training_generator):
            optimizer.zero_grad()
            samples.requires_grad = True
            divs = laplacian(samples, model.forward, keep_graph=True, create_graph=True, return_grad=True)
            term1 = divs[0]
            term1 = term1 * b ** -1
            vals = divs[2]
            vals2 = mp.mueller_grad_vectorized_torch(samples)
            term2 = torch.zeros(len(term1))
            for j in range(len(vals)):
                term2[j] = torch.dot(vals[j], vals2[j])

            outputs = term1 + term2
            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()
            # Backward and optimize

            if (epoch + 1) % 2 == 0 and i % 3 == 0:
                # arr = samples.detach().numpy()
                # plt.scatter(X,Y,marker='o',s=1)
                # plt.scatter(arr[:,0], arr[:,1],marker='o',s=10, c='red')
                #
                # plt.show()
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


def laplacian(xs, f, create_graph=False, keep_graph=None, return_grad=False):
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    ys = f(xs_flat.view_as(xs))
    (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
    ones = torch.ones_like(ys_g)
    (dy_dxs,) = torch.autograd.grad(ys_g, xs_flat, ones, create_graph=True)
    lap_ys = sum(
        torch.autograd.grad(
            dy_dxi, xi, ones, retain_graph=True, create_graph=create_graph
        )[0]
        for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
    )
    if not (create_graph if keep_graph is None else keep_graph):
        ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
    result = lap_ys, ys
    if return_grad:
        result += (dy_dxs.detach().view_as(xs),)
    return result

# X = np.zeros(n)
# Y = np.zeros(n)
# for i in tqdm(range(n)):
#     x = mp.getNextIteration(x, h, updaters, sigma=sigma, omega=omega, b=b)
#     X[i] = x[0]
#     Y[i] = x[1]
# plt.scatter(X, Y)

def test(x):
    return x[:,0]**2 + 10*x[:,0]*x[:,1] + 3*x[:,1]**2

def grad_test(x):
    ret = np.zeros_like(x)
    term1 = 2*x[:,0] + 10*x[:,1]
    term2 = 10*x[:,0] + 6*x[:,1]
    ret[:,0] = term1
    ret[:,1] = term2
    return ret

if __name__ == "__main__":
    # arr = np.zeros((10,2))
    # for i in range(len(arr)):
    #     arr[i] = np.random.randn(2) * 10
    # real_grad = grad_test(arr)
    # torch_arr = torch.from_numpy(arr).float()
    # torch_arr.requires_grad = True
    # print(torch_arr)
    #
    # lapl = laplacian(torch_arr, test,return_grad=True)
    main()