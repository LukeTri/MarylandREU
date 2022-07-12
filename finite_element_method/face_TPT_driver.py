#!/usr/bin/env python
# coding: utf-8

# In[1]:


from distmesh import *
from FEM_TPT import *
import numpy as np
import math
import matplotlib.pyplot as plt
import csv 


# In[2]:


# finds the committor, the reactive current, and the reaction rate for the face potential 
# using the finite element method and the distmesh triangulation

# the committor equation 

# \nabla \cdot (\exp( - \beta fpot(x) \nabla q(x) ) = 0, x \in (A\cup B)^c
# q(\partial A) = 0, q(\partial B) = 1
# dq/dn = 0, x \in outer boundaty defined by {x : fpot(x) = Vbdry}

# the homogeneous Neumann boundary condition dq/dn = 0 means that 
# the trajectory reflects from the outer boundary whenever it reaches it


# In[3]:


# parameters for the face potential
xa=-3 
ya=3
xb=0 
yb=4.5
par = np.array([xa,ya,xb,yb]) # centers of sets A and B

# parameters for the mueller potential
xa=0.62
ya=0.03
xb=-0.55
yb=1.44
par = np.array([xa,ya,xb,yb]) # centers of sets A and B

# problem setup: choose sets A and B and the outer boundary
# set A is the circle with center at (xa,ya) and radius ra
# set B is the circle with center at (xb,yb) and radius rb
ra = 0.1 # radius of set A
rb = 0.1 # radius of set B
beta = 1/30 # beta = 1/(k_B T), T = temperature, k_B = Boltzmann's constant
Vbdry = 50 # level set of the outer boundary {x : fpot(x) = Vbdry}
neg_bdry = -150

# if generate_mesh = True, mesh is generated and saves as csv files
# if generate_mesh = False, mesh is downloaded from those csv files
generate_mesh = True

# h0 is the desired scalind parameter for the mesh
h0 = 0.02


def face2(xy, a=np.array([-1, -1, -6.5, 0.7]), b=np.array([0, 0, 11, 0.6]),
                                  c=np.array([-10, -10, -6.5, 0.7]),
                                  d=np.array([-200, -100, -170, 15]), z=np.array([1, 0, -0.5, -1]),
                                  Y=np.array([0, 0.5, 1.5, 1])):
    ret = 0
    for i in range(0, 4):
        ret += d[i] * np.e ** (a[i] * (xy[:,0] - z[i]) ** 2 + b[i] * (xy[:,0] - z[i]) *
                               (xy[:,1] - Y[i]) + c[i] * (xy[:,1] - Y[i]) ** 2)
    return ret


def face(xy):
    x = xy[:,0]
    y = xy[:,1]
    f=(1-x)**2+(y-0.25*x**2)**2+1
    g1=1-np.exp(-0.125*((x-xa)**2+(y-ya)**2))
    g2=1-np.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
    g3=1.2-np.exp(-2*((x+0)**2+(y-2)**2))
    g4=1+np.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
    v = f*g1*g2*g3*g4
    return v

# define face potential on a meshgrid
nx,ny= (100,100)
nxy = nx*ny
xmin = -4.8
xmax = 4.2
ymin = -3
ymax = 6
x1 = np.linspace(xmin,xmax,nx)
y1 = np.linspace(ymin,ymax,ny)
x_grid, y_grid = np.meshgrid(x1,y1)
x_vec = np.reshape(x_grid, (nxy,1))
y_vec = np.reshape(y_grid, (nxy,1))
v = np.zeros(nxy)
xy = np.concatenate((x_vec,y_vec),axis=1)
v = face2(xy)
vmin = np.amin(v)
v_grid = np.reshape(v,(nx,ny))    
# graphics
plt.rcParams.update({'font.size': 20})
ls = plt.contour(x_grid,y_grid,v_grid,np.arange(neg_bdry,Vbdry,20))
plt.colorbar(label="Potential function", orientation="vertical")
axes=plt.gca()
axes.set_aspect(1)


# In[4]:


# set sets A and B and the outer boundary
Na = int(round(2*math.pi*ra/h0))
Nb = int(round(2*math.pi*rb/h0))
ptsA = put_pts_on_circle(xa,ya,ra,Na)
ptsB = put_pts_on_circle(xb,yb,rb,Nb)

# outer boundary
bdrydata = plt.contour(x_grid,y_grid,v_grid,[Vbdry]) # need this for the meshing
for item in bdrydata.collections:
    for i in item.get_paths():
        p_outer = i.vertices
# reparametrize the outer boundary to make the distance 
# between the nearest neighbor points along it approximately h0
pts_outer = reparametrization(p_outer,h0);

Nouter = np.size(pts_outer,axis=0)
Nfix = Na+Nb+Nouter

plt.scatter(pts_outer[:,0],pts_outer[:,1],s=10)
plt.scatter(ptsA[:,0],ptsA[:,1],s=10)
plt.scatter(ptsB[:,0],ptsB[:,1],s=10)
axes=plt.gca()
axes.set_aspect(1)
plt.rcParams.update({'font.size': 20})
plt.show()


# In[5]:


# input data for triangulation
# bbox = [xmin,xmax,ymin,ymax]
if generate_mesh == True:
    bbox = [xmin,xmax,ymin,ymax]
    pfix = np.zeros((Nfix,2))
    pfix[0:Na,:] = ptsA
    pfix[Na:Na+Nb,:] = ptsB
    pfix[Na+Nb:Nfix,:] = pts_outer

    def dfunc(p):
        d0 = face2(p)
        dA = dcircle(p,xa,ya,ra)
        dB = dcircle(p,xb,yb,rb)
        d = ddiff(d0-Vbdry,dunion(dA,dB))
        return d

    pts,tri = distmesh2D(dfunc,huniform,h0,bbox,pfix)
    with open('face_pts.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(pts)

    with open('face_tri.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(tri)
else:
    pts = np.loadtxt('face_pts.csv', delimiter=',', dtype=float)
    tri = np.loadtxt('face_tri.csv', delimiter=',', dtype=int)

Npts = np.size(pts,axis=0)
Ntri = np.size(tri,axis=0)
print("Npts = ",Npts," Ntri = ",Ntri)    


# In[6]:


# find the mesh points lying on the Dirichlet boundary \partial A \cup \partial B
NAind,Aind = find_ABbdry_pts(pts,xa,ya,ra,h0) # find mesh points on \partial A
NBind,Bind = find_ABbdry_pts(pts,xb,yb,rb,h0) # find mesh points on \partial B

def fpot(pts):
    return face2(pts)

# find the committor
q = FEM_committor_solver(pts,tri,Aind,Bind,fpot,beta)

TPTdata = np.concatenate((pts,np.reshape(q,(Npts,1))),axis = 1)
with open('../data/fe_mueller_b=0.033.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(TPTdata)
