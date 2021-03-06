import time

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
import autograd.numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from descartes import PolygonPatch
import alphashape
from tqdm import tqdm

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


def createGraph(x, h, n, plot_row, plot_col, update_step_size=1000, gaussian=True, sigma=0.05, omega=5):
    updaters = []
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in tqdm(range(n)):
        if (i % update_step_size) == update_step_size-1 and gaussian:
            updaters.append(x)
        x = mp.getNextIteration(x, h, updaters=updaters, offset_func="metadynamics", sigma=sigma, omega=omega)
        X[i] = x[0]
        Y[i] = x[1]

    for i in range(len(updaters)):
        ax.plot(updaters[i][0], updaters[i][1], markersize=20, marker="o")
    ax.scatter(X, Y)

    alpha_shape = alphashape.alphashape(np.array([*zip(X, Y)]), 0)

    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4))

    plot_countours()

    end = time.time()
    print(end - start)


createGraph(np.array([0, 0]), 10 ** -5, 100000, 0, 0, omega=5)

ax.title.set_text('omega=5,time_step=1000')

plt.show()
