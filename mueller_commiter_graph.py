import colorsys
import mueller_potential as mp
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def mapProbabilities(grid_size, num_minima, x_start, x_stop, y_start, y_stop, trials, delta, h):
    color_vals = np.zeros((grid_size, grid_size))
    X = np.zeros(grid_size*grid_size)
    Y = np.zeros(grid_size*grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            X[i*grid_size + j] = x_start + i * (x_stop - x_start) / grid_size
            Y[i * grid_size + j] = y_start + j * (y_stop - y_start) / grid_size
            x = np.array([x_start + i * (x_stop - x_start) / grid_size, y_start + j * (y_stop - y_start) / grid_size])
            vals = getProbability(x, trials, delta, h)
            for p in range(num_minima):
                color_vals[i][j] = color_vals[i][j] + vals[p] * p / (num_minima)

    Z = np.ravel(color_vals, 'F')
    print(X)
    print(Y)
    print(Z)
    HSV = [(Z[x], 1, 1) for x in range(len(Z))]
    RGB = np.array([colorsys.hsv_to_rgb(*x) for x in HSV])

    plt.scatter(X, Y, c=RGB)
    plt.show()



def getFirstMinimum(x, h, delta):
    updaters = []
    while True:
        x = mp.getNextIteration(x, h, updaters)
        for i in range(3):
            if np.linalg.norm(x - mp.mumins[i]) < delta:
                return i


def getProbability(x, n, delta, h):
    vals = np.zeros(3)
    for i in range(n):
        vals[getFirstMinimum(x, h, delta)] += 1
    vals = vals/n
    return vals


mapProbabilities(10, 3, -1.5, 1.5, -0.5, 2, 2, 0.5, 10**-5)