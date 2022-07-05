import time

from euler_maruyama import euler_maruyama_white_noise as mp
import autograd.numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from tqdm import tqdm

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

def plot_contours():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))
    plt.figure()
    Z = v_func(X, Y)
    tics = np.linspace(-150, 150, 30)
    CS = plt.contour(X, Y, Z, tics)
    plt.clabel(CS, inline=False, fontsize=10)

def convert_q_to_colors(q):
    c = np.zeros((len(q),3))
    for i in range(len(q)):
        c[i][0] = q[i]
        c[i][1] = 1 - q[i]
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
    return q, c_bool


def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=5):
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

    plot_contours()
    q, c_bool = eig_stuff(newX, newY)

    newX_C = newX[c_bool]
    newY_C = newY[c_bool]

    c = convert_q_to_colors(q)

    plt.scatter(newX_C, newY_C,s=100, c=c)
    plt.scatter(X, Y, s=1)

    end = time.time()
    print(end - start)


createGraph(np.array([0, 0]), 10 ** -5, 100000, 0, 0, omega=5)

plt.show()
