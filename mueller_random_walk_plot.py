import mueller_potential as mp
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt


def createGraph(x, h, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in range(n):
        x = mp.getNextIteration(x, h)
        X[i] = x[0]
        Y[i] = x[1]

    plt.scatter(X, Y)
    plt.show()


createGraph(np.array([0, 0]), 10 ** -5, 100000)
