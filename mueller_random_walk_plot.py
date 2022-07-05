import time

from matplotlib.gridspec import GridSpec

import mueller_potential as mp
import autograd.numpy as np
from autograd import grad
from autograd import jacobian
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn')
import pandas as pd
from descartes import PolygonPatch
import alphashape
from numpy import linalg as npl
from tqdm import tqdm
import csv

updaters = []
fig, ax = plt.subplots()


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_bounding_area(alpha_shape):
    bound_x, bound_y = alpha_shape.exterior.coords.xy
    return PolyArea(bound_x, bound_y)



def get_updated_offset(x, y, updaters, omega=5, sigma=0.05):
    offset = 0
    for q in range(len(updaters)):
        xn1 = updaters[q][0]
        xn2 = updaters[q][1]
        offset += omega * np.e ** (-((x - xn1)**2 + (y - xn2)**2)/sigma)
    return offset

def plot_countours():
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))
    Z = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            print()
            Z[j][i] = get_updated_offset(X[0][i], Y[j][0], updaters)
    tics = np.linspace(-150, 150, 30)
    CS = plt.contour(X, Y, Z, tics)
    plt.clabel(CS, inline=False, fontsize=10)

def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=20, b = 1/20):
    updaters = []
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        # if (i % update_step_size) == update_step_size-1 and gaussian:
            # updaters.append(x)
        x = mp.getNextIteration(x, h, updaters,sigma=sigma, omega=omega, b=b)
        X[i] = x[0]
        Y[i] = x[1]
    ax.scatter(X, Y)

    alpha_shape = alphashape.alphashape(np.array([*zip(X, Y)]), 0)

    # z1, f1, evaluations, h = gd(f, g, z_0, step_function, state)

    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4))

    end = time.time()
    print(end - start)
    return X,Y

mp.plot_contours()

X, Y = createGraph(np.array([0, 0]), 10 ** -5, 500000, 0, 0, omega=5,b=1/30)

header = ['X', 'Y']
data = np.vstack((X,Y)).T
print(np.shape(data))
data = np.ndarray.tolist(data)
print(len(data))

with open('data/mueller_standard_b=0.033_n=500000.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)

ax.title.set_text('omega=5,time_step=1000')

plt.show()
