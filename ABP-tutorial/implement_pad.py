import numpy as np
from math import sin, cos, sqrt, pi
from random import uniform
from random import gauss
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Particle:
    def __init__(self, id, position, theta, velocity=[0.0, 0.0], force=[0.0, 0.0]):
        self.id = id
        self.position = np.array(position)
        self.old_position = np.array(position)  # Store old position in particle
        self.theta = theta
        self.direction = np.array([cos(theta), sin(theta)])
        self.velocity = np.array(velocity)
        self.force = np.array(force)

class ParticleSimulation:
    def __init__(self, lx=20, ly=20, density=0.8, a=1, v0=1, noise_amplitude=pi/4,
                 k=1, J=1, gamma_t=1, gamma_rot=1, Dr=1, Dt=1, dt=0.1):
        # System parameters
        self.lx, self.ly = lx, ly
        self.xmin, self.xmax = -0.5*lx, 0.5*lx
        self.ymin, self.ymax = -0.5*ly, 0.5*ly
        
        # Particle parameters
        self.N = int(lx*ly*density)
        self.a = a  
        self.rcut = 3*a
        self.pad = 0.5*a
        self.v0 = v0
        
        # Physical parameters
        self.noise_amplitude = noise_amplitude
        self.k = k
        self.J = J
        self.gamma_t = gamma_t
        self.gamma_rot = gamma_rot
        self.dt = dt
        self.Dr = Dr
        self.Dt = Dt
        
        # Initialize particles and neighbor list
        self.particles = self.initialize_particles()
        self.neighbors_list = [[] for _ in range(self.N)]
        self.needs_rebuild = True  # Flag for neighbor list rebuild
        
    def initialize_particles(self):
        particles = []
        for i in range(self.N):
            position = [uniform(self.xmin, self.xmax), uniform(self.ymin, self.ymax)]
            theta = uniform(-pi, pi)
            particles.append(Particle(i, position, theta))
        return particles
    
    def check_rebuild(self):
        """Check if any particle has moved more than half the padding distance"""
        for particle in self.particles:
            dr = particle.position - particle.old_position
            if np.linalg.norm(dr) >= 0.5 * self.pad:
                return True
        return False

    def update_neighbors(self):
        """Update neighbor list and store current positions"""
        self.neighbors_list = [[] for _ in range(self.N)]
        for i in range(self.N):
            # Update old position when rebuilding neighbor list
            self.particles[i].old_position = np.array(self.particles[i].position)
            for j in range(self.N):
                if i != j:
                    dist = np.linalg.norm(self.particles[i].position - self.particles[j].position)
                    if dist < self.rcut + self.pad:
                        self.neighbors_list[i].append(j)
        self.needs_rebuild = False
    
    def harmonic_force(self, d):
        return self.k * (2*self.a - d)
    
    def calculate_forces(self, i):
        particle = self.particles[i]
        force = np.zeros(2)
        torque = 0
        
        for j in self.neighbors_list[i]:
            neighbor = self.particles[j]
            diff = particle.position - neighbor.position
            d = np.linalg.norm(diff)
            
            if d < 2*self.a:
                force_magnitude = self.harmonic_force(d)
                force += force_magnitude * (diff / d)
                
            torque += -self.J * sin(particle.theta - neighbor.theta)
        
        return force, torque
    
    def update_particle(self, i):
        particle = self.particles[i]
        
        # Check if we need to rebuild neighbor list
        if not self.needs_rebuild:
            self.needs_rebuild = self.check_rebuild()
            
        force, torque = self.calculate_forces(i)
        
        # Add thermal noise
        thermal_noise_rot = sqrt(2*self.Dr*self.dt)*gauss(0,1)
        thermal_noise_trans = sqrt(2*self.Dt*self.dt)*gauss(0,1)
        
        # Update orientation
        particle.theta += (torque/self.gamma_rot) * self.dt + thermal_noise_rot
        particle.direction = np.array([cos(particle.theta), sin(particle.theta)])
        
        # Update velocity and position
        noise_vector = np.array([gauss(0,1), gauss(0,1)]) * thermal_noise_trans
        particle.velocity = (self.v0 * particle.direction + force/self.gamma_t) + noise_vector
        particle.position += particle.velocity * self.dt
        
        # Apply periodic boundary conditions
        particle.position[0] = ((particle.position[0] + self.lx/2) % self.lx) - self.lx/2
        particle.position[1] = ((particle.position[1] + self.ly/2) % self.ly) - self.ly/2
    
    def simulate(self):
        fig, ax = plt.subplots()
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_aspect('equal')
        ax.set_title('Particle Dynamics')
        
        positions = np.array([p.position for p in self.particles])
        directions = np.array([p.direction for p in self.particles])
        
        scatter = ax.scatter(positions[:, 0], positions[:, 1], s=30, c='blue')
        quiver = ax.quiver(positions[:, 0], positions[:, 1], 
                          directions[:, 0], directions[:, 1], 
                          scale=25)
        
        def animate(frame):
            # Update neighbor list if needed
            if self.needs_rebuild:
                self.update_neighbors()
            
            for i in range(self.N):
                self.update_particle(i)
            
            positions = np.array([p.position for p in self.particles])
            directions = np.array([p.direction for p in self.particles])
            
            scatter.set_offsets(positions)
            quiver.set_offsets(positions)
            quiver.set_UVC(directions[:, 0], directions[:, 1])
            
            return scatter, quiver
        
        ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
        plt.show()
        return ani

# Create and run simulation
sim = ParticleSimulation()
sim.simulate()