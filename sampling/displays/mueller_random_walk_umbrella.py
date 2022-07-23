import time

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
import autograd.numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from tqdm import tqdm

updaters = []
fig, ax = plt.subplots()

start_x = -5
end_x = 5
start_y = -5
end_y = 5


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_bounding_area(alpha_shape):
    bound_x, bound_y = alpha_shape.exterior.coords.xy
    return PolyArea(bound_x, bound_y)


def get_updated_offset(x, c, d, z):

    alpha = -(c[0] * x[0] + x[1] * c[1] + d) / (c[0]**2 + c[1] ** 2)

    if x[1] + c[1] * alpha - z[1] > 0:
        return -np.sqrt((x[0] + c[0] * alpha - z[0])**2 + (x[1] + c[1] * alpha - z[1])**2)
    return np.sqrt((x[0] + c[0] * alpha - z[0])**2 + (x[1] + c[1] * alpha - z[1])**2)

def get_updated_offset2(x, c, d, z):
    return z - c[0] * x[0] - c[1] * x[1]


def affine(c, d, x):
    return (d-c[1] * x)/ c[0]


def createGraph(x, h, n, plot_row, plot_col, k=1000,c=np.array([-2,1]), d=0, z=np.array([0,0])):
    updaters = []
    start = time.time()
    X = np.zeros(n)
    Y = np.zeros(n)

    rie_sum = 0

    for i in tqdm(range(n)):
        x = mp.getNextIteration(x, h, offset_func="umbrella", updaters=updaters, k=k,c=c,d=d,z=z)
        rie_sum += get_updated_offset2(x, c, d, z)
        X[i] = x[0]
        Y[i] = x[1]

    ax.scatter(X, Y)


    # alpha_shape = alphashape.alphashape(np.array([*zip(X, Y)]), 0)

    # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4))

    end = time.time()
    print(end - start)
    return rie_sum * 10**-5 * k


def plotMuellerContours2():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(start_x, end_x, 100),
                       np.linspace(start_y, end_y, 100))
    Z = v_func(X, Y)
    tics = np.array([-150, -125, -100, -75, -50, -25, 0,10, 40, 60,100,1000,10000,100000, 1000000])
    CS = ax.contour(X, Y, Z, tics,colors='black',alpha=0.5)
    ax.clabel(CS, inline=False, fontsize=10)


def get_forces(x, h, n, plot_row, plot_col, k=1000,c=np.array([-2,1]), d=0, z=np.array([0,0])):
    forces = np.zeros(50)
    intercepts = np.linspace(-2,0.5,50)
    for i in range(50):
        forces[i] = createGraph(np.array([0,0]), h, n, plot_row, plot_col, k=k,c=c, d=d, z = intercepts[i])
    plotMuellerContours2()
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(intercepts,forces)
    print(forces)
    print(intercepts)


get_forces(np.array([0, 0]), 10 ** -5, 5000, 0, 0, k=10000,c=np.array([1/2,-1]),d=0,z=np.array([0,1]))


ax.title.set_text('omega=5,time_step=1000')

plt.show()
