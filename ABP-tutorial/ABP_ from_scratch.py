import numpy as np
from math import sin, cos, sqrt, pi
from random import uniform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
#Box
lx = 10
ly = 10
xmin, xmax = -0.5*lx, 0.5*lx 
ymin, ymax = -0.5*ly, 0.5*ly

# Particles
density = 0.4
N = int(lx*ly*density)
a = 1
rcut = 2*a
v0 = 1

#physical_parameters
noise_amplitude = pi/4
k = 1
J = 1
gamma_t = 1
gamma_rot = 1
dt = 0.1
# List of particles : [id,r,n,v,f]
particles = []

for i in range(N):
    id = i
    r = [uniform(xmin, xmax),uniform(ymin, ymax )]
    theta = uniform(-pi, pi)
    n = [cos(theta), sin(theta)]
    v = [0.0, 0.0]
    f = [0.0, 0.0]
    particles.append([id,r,theta,n,v,f])

# Neigbors
# Doing like in the tutorial I will use a r_cutoff and a pad distance to update the list only every 10 iters 
neighbors_list = []
def update_neighbors():
    for i in range(N):
        sub = particles[i]
        rx,ry =  sub[1]
        neighbors = []
        for j in range(N):
            target = particles[j]
            rx2,ry2 = target[1]
            if sqrt((rx-rx2)**2 + (ry - ry2)**2) < rcut:
                neighbors.append(target[0])
                print(target[0])
        neighbors_list.append(neighbors)

print(neighbors_list)

def noise():
    return noise_amplitude*uniform(-1,1)
# Simulation 

def harmonic_force(d):
    return k*(2*a-d)

def force_and_torque(i):
    part = particles[i] 
    rx,ry = part[1]
    tau = 0
    fx = 0
    fy = 0
    for neigbor in neighbors_list[i]:
        rx2,ry2 = neigbor[1]
        d = sqrt((rx-rx2)**2 + (ry - ry2)**2)
        rij = [(rx+rx2)/d,(ry+ry2)/d]

        if d < 2*a:
            F = harmonic_force(d)
            fx += rij[0]*F
            fy += rij[1]*F
        tau += -J*sin(part[2]-neigbor[2])
    force = [fx,fy]
    return force,tau

def update(i):
    part = particles[i]
    x,y = part[1]
    theta = part[2]
    nx,ny = part[3]
    vx,vy = part[4]
    fx,fy = part[5]
    force,tau = force_and_torque(i)
    theta = tau
    vx = vx*cos(theta)
    vy = vy*sin(theta)
    vx += fx
    vy += fy
    x = x + vx
    y = y + vy
    part[1] = x,y
    part[2] = theta
    
#simulation
fig, ax = plt.subplots()
ax.set_xlim(-lx / 2, lx / 2)
ax.set_ylim(-ly / 2, ly / 2)
ax.set_aspect('equal')
ax.set_title('Particle Dynamics')

# Create plot objects for particles
scatter = ax.scatter([], [], s=30, c='blue')

def animate(frame):
    if frame%10:
        update_neighbors()  
    for i in range(N):
        update(i)

    positions = np.array([part[1] for part in particles])
    scatter.set_offsets(positions)  # Update scatter plot with new positions
    return scatter,

ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()
