import numpy as np
from math import sin, cos, sqrt, pi
from random import uniform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
# Box
lx = 10
ly = 10
xmin, xmax = -0.5 * lx, 0.5 * lx
ymin, ymax = -0.5 * ly, 0.5 * ly

# Particles
density = 0.4
N = int(lx * ly * density)
a = 1
rcut = 2 * a
v0 = 1

# Physical parameters
noise_amplitude = pi / 4
k = 1
J = 1
gamma_t = 1
gamma_rot = 1

# List of particles: [id, r, n, v, f]
particles = []

for i in range(N):
    id = i
    r = [uniform(xmin, xmax), uniform(ymin, ymax)]
    theta = uniform(-pi, pi)
    n = [cos(theta), sin(theta)]
    v = [0.0, 0.0]
    f = [0.0, 0.0]
    particles.append([id, r, theta, n, v, f])

# Neighbors
neighbors_list = []

def update_neighbors():
    global neighbors_list
    neighbors_list = []
    for i in range(N):
        sub = particles[i]
        rx, ry = sub[1]
        neighbors = []
        for j in range(N):
            target = particles[j]
            rx2, ry2 = target[1]
            if sqrt((rx - rx2)**2 + (ry - ry2)**2) < rcut:
                neighbors.append(j)
        neighbors_list.append(neighbors)

def noise():
    return noise_amplitude * uniform(-1, 1)

# Force and torque calculation
def harmonic_force(d):
    return k * (2 * a - d)

def force_and_torque(i):
    part = particles[i]
    rx, ry = part[1]
    tau = 0
    fx = 0
    fy = 0
    for neigbor_id in neighbors_list[i]:
        neighbor = particles[neigbor_id]
        rx2, ry2 = neighbor[1]
        d = sqrt((rx - rx2) ** 2 + (ry - ry2) ** 2)
        rij = [(rx - rx2) / d, (ry - ry2) / d]
        if d < 2 * a:
            F = harmonic_force(d)
            fx += rij[0] * F
            fy += rij[1] * F
        tau += -J * sin(part[2] - neighbor[2])
    force = [fx, fy]
    return force, tau

# Update particle positions and velocities
def update(i):
    part = particles[i]
    x, y = part[1]
    theta = part[2]
    nx, ny = part[3]
    vx, vy = part[4]
    fx, fy = part[5]
    
    force, torque = force_and_torque(i)
    
    # Update theta and velocity components
    theta = (torque + noise()) % (2 * pi)
    vx = (v0 * nx + (1 / gamma_t) * force[0] + noise()) * cos(theta)
    vy = (v0 * ny + (1 / gamma_t) * force[1] + noise()) * sin(theta)
    
    # Update position with periodic boundary conditions
    x, y = [(x + vx) % lx, (y + vy) % ly]
    part[1] = x, y
    part[2] = theta
    part[4] = vx, vy

# Visualization and animation setup
fig, ax = plt.subplots()
ax.set_xlim(-lx / 2, lx / 2)
ax.set_ylim(-ly / 2, ly / 2)
ax.set_aspect('equal')
ax.set_title('Particle Dynamics')

# Create plot objects for particles
scatter = ax.scatter([], [], s=30, c='blue')

# Update function for animation
def animate(frame):
    update_neighbors()  # Update neighbors list every frame
    for i in range(N):
        update(i)
    
    # Extract particle positions for plotting
    positions = np.array([part[1] for part in particles])
    scatter.set_offsets(positions)  # Update scatter plot with new positions
    return scatter,  # Return tuple for FuncAnimation

# Create the animation (without blit=True)
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=False)

# Show the animation
plt.show()
