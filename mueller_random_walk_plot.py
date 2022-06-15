import mueller_potential as mp
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn')
updaters = []


def createGraph(x, h, n, update_step_size=1000):
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in range(n):
        if (i % update_step_size) == update_step_size-1:
            updaters.append(x)
        x = mp.getNextIteration(x, h, updaters)
        X[i] = x[0]
        Y[i] = x[1]

    for i in range(len(updaters)):
        plt.plot(updaters[i][0], updaters[i][1], markersize=20, marker="o")
    print(updaters)
    plt.scatter(X, Y)
    plt.show()


createGraph(np.array([0, 0]), 10 ** -5, 10000)
