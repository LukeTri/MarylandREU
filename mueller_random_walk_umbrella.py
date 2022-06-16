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

updaters = []
fig, ax = plt.subplots()

start_x = -5
end_x = 3
start_y = -2
end_y = 3


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


def affine(c, d, x):
    return (d-c[1] * x)/ c[0]


def createGraph(x, h, n, plot_row, plot_col, k=1000,c=np.array([-2,1]), d=0, z=np.array([0,0])):
    updaters = []
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        x = mp.getNextIteration(x, h, offset_func="umbrella", updaters=updaters, k=k,c=c,d=d,z=z)
        X[i] = x[0]
        Y[i] = x[1]

    ax.scatter(X, Y)
    new_x = np.linspace(start_x,end_x,10)
    new_y = np.zeros(10)
    for x in range(len(new_x)):
        new_y[x] = affine(c, d, new_x[x])
    ax.plot(new_x, new_y)

    # alpha_shape = alphashape.alphashape(np.array([*zip(X, Y)]), 0)

    # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4))

    plotMuellerContours2()

    end = time.time()
    print(end - start)


def plotMuellerContours2():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(start_x, end_x, 100),
                       np.linspace(start_y, end_y, 100))
    Z = v_func(X, Y)
    tics = np.array([-150, -125, -100, -75, -50, -25, 0,10, 40, 60,100,1000,10000,100000])
    CS = ax.contour(X, Y, Z, tics,colors='green')
    ax.clabel(CS, inline=False, fontsize=10)


def get_forces(x, h, n, plot_row, plot_col, k=1000,c=np.array([-2,1]), d=0):
    for i in range(5):
        createGraph(np.array([0,i]), h, n, plot_row, plot_col, k=k,c=c, d=d)


createGraph(np.array([0, 0]), 10 ** -5, 2000, 0, 0, k=10000,c=np.array([1,1]),d=0,z=np.array([-1,1]))

ax.title.set_text('omega=5,time_step=1000')

plt.show()
