import mueller_potential as mp
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn')
updaters = []

def getWalkData(x, n, h):
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in range(n):
        x = mp.getNextIteration(x, h, updaters)
        X[i] = x[0]
        Y[i] = x[1]
    return X, Y

#general figure options
n = 1000
step_size = 100

fig = plt.figure(figsize=(15, 7))
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(- 0.5, 2))
line, = ax.plot([], [], lw=2)
ax.set_title('2D Random Walk', fontsize=18)
ax.set_xlabel('x_val', fontsize=16)
ax.set_ylabel('y_val', fontsize=16)
ax.tick_params(labelsize=12)
X, Y = getWalkData(np.array([0,0]), n, 10**-5)

# initialization function
def init():
    # creating an empty plot/frame
    line.set_data([], [])
    return line,

xdata, ydata = [], []

# animation function with large gaps between points
def animate1(i):
    x = X[i * step_size]
    y = Y[i * step_size]

    xdata.append(x)
    ydata.append(y)
    line.set_data(xdata, ydata)
    return line,

# animation function without large gaps between points
def animate2(i):
    for j in range(step_size):
        x = X[i * (step_size-1) + j]
        y = Y[i * (step_size-1) + j]

        xdata.append(x)
        ydata.append(y)
    line.set_data(xdata, ydata)
    return line,

anim = animation.FuncAnimation(fig, animate2, init_func=init, frames=n//step_size, interval=100, blit=True)
anim.save('mueller_walk.gif',writer='imagemagick')