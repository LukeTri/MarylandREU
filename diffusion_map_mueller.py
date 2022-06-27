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
sigma = 0.2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

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
    epsilon = 10

    k = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            k[i][j] = np.exp(-(dist[i][j]) ** 2 / epsilon)

    q = np.zeros(len(X))
    P = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        q[i] = np.sum(k[i])
        P[i] = k[i] / q[i]

    w, v = np.linalg.eig(P)

    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    v_tran = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            v_tran[i][j] = v[j][i]

    pi = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        pi[i][i] = q[i] / np.sum(q)

    scaled_identity = v_tran @ pi @ v

    for i in range(len(X)):
        for j in range(len(X)):
            v[i][j] = v[i][j] / np.sqrt(scaled_identity[j][j])
            v_tran[j][i] = v_tran[j][i] / np.sqrt(scaled_identity[j][j])

    newX = np.zeros((len(X), 3))

    indices = np.array([1, 2, 3])
    print(w[indices[0]] - w[indices[-1]])
    for i in range(len(X)):
        newX[i] = diffusion_map(i, w, v, indices)

    x = newX[:, 0]
    y = newX[:, 1]
    z = newX[:, 2]
    ax.scatter(x, y, z)
    plt.show()

    return newX[:, 0]

def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=20):
    updaters = []
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        x = mp.getNextIteration(x, h, updaters,sigma=sigma, omega=omega)
        X[i] = x[0]
        Y[i] = x[1]

    step_size = 100
    newX = np.zeros(len(X) // step_size)
    newY = np.zeros(len(Y) // step_size)
    for i in range(len(newX)):
        newX[i] = X[i * step_size]
        newY[i] = Y[i * step_size]

    colors = eig_stuff(newX, newY)
    plt.scatter(newX, newY, c=colors)

    end = time.time()
    print(end - start)


createGraph(np.array([0, 0]), 10 ** -5, 100000, 0, 0, omega=5)

ax.title.set_text('omega=5,time_step=1000')

plt.show()
