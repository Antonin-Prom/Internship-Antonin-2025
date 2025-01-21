#Brute_force
import numpy as np
from math import sin, cos, sqrt, pi
from random import uniform
from random import gauss
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import vtk

class Particle:
    def __init__(self, id, position, theta, velocity=[0.0, 0.0], force=[0.0, 0.0]):
        self.id = id
        self.position = np.array(position)
        self.theta = theta
        self.direction = np.array([cos(theta), sin(theta)])
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.tau = 0

class ParticleSimulation:
    def __init__(self, lx=50, ly=50, density=0.4, a=1, v0=1, noise_amplitude=0.02*pi,
                 k=10, J=2, gamma_t=1, gamma_rot=1, Dr=1, Dt= 0, dt=0.1):
        
        #Box parameters
        self.lx, self.ly = lx, ly
        self.xmin, self.xmax = -0.5*lx, 0.5*lx
        self.ymin, self.ymax = -0.5*ly, 0.5*ly

        # Particle parameters
        self.N = int(lx*ly*density)
        self.a = a  
        self.rcut = 5*a
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

        #Initalise
        self.particles = []
        self.neighbors = [[] for _ in range(self.N)]

    def initialise_file(self,outfile = 'raw.json'):
        particles = []
        N = self.N
        lx = self.lx
        ly = self.ly
        for i in range(N):
            r = [uniform(self.xmin, self.xmax), uniform(self.ymin, self.ymax)]
            theta = uniform(-pi, pi)
            particles.append({'r': r, 'theta': theta})
        jsonData = {}
        jsonData["system"] = {}
        jsonData["system"]["Number"] = {"N":N}
        jsonData["system"]["box"] = {"Lx": lx, "Ly": ly}
        jsonData["system"]["particles"] = particles
        with open(outfile, 'w') as out:
            json.dump(jsonData, out, indent = 4)

    def read_init_config(self, initfile = 'raw.json'):
        with open(initfile) as f:
            self.particles = []
            data = json.load(f)
            Lx = data["system"]["box"]["Lx"]
            Ly = data["system"]["box"]["Ly"]
            self.lx,self.ly = Lx, Ly
            id = 0
            for p in data['system']['particles']:
                x, y = p['r']
                theta = p['theta']
                self.particles.append(Particle(id=id,position=[x,y],theta=theta,velocity=[0.0, 0.0], force=[0.0, 0.0]))
        
    def build_neighbor(self):
        self.neighbors = [[] for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    dist = np.linalg.norm(self.particles[i].position - self.particles[j].position)
                    if dist <= self.rcut: 
                        self.neighbors[i].append(j)         

    def harmonic_force(self, d):
        return self.k * (2*self.a - d)     

    def calculate_force_torque(self, i):
        particle = self.particles[i]
        force = np.zeros(2)
        torque = 0
        xi = particle.position[0]
        yi = particle.position[1]
        for j in self.neighbors[i]:
            neighbor = self.particles[j]
            diff = particle.position - neighbor.position
            d = np.linalg.norm(diff)
            xj = neighbor.position[0]
            yj = neighbor.position[1]
            if d < 2*self.a:
                force_magnitude = self.harmonic_force(d)
                force += force_magnitude * (diff / d)
                
            tau_z = self.J*(xi*yj - yi*xj)
            particle.tau += tau_z 
        return force,particle.tau
    
    def update_particle(self,i):     
        particle = self.particles[i]
        force, torque = self.calculate_force_torque(i)
        #thermal_noise_rot = sqrt(2*self.Dr*self.dt)*gauss(0,1)
        thermal_noise_rot = uniform(-self.noise_amplitude,+self.noise_amplitude)
        thermal_amplitude_trans = sqrt(2*self.Dt*self.dt)

        particle.theta += (torque/self.gamma_rot) * self.dt + thermal_noise_rot
        particle.direction = np.array([cos(particle.theta), sin(particle.theta)])
        
        # Update velocity and position
        noise_vector = np.array([gauss(0,1), gauss(0,1)]) * thermal_amplitude_trans
        particle.velocity = (self.v0 * particle.direction + force/self.gamma_t) + noise_vector
        particle.position += particle.velocity * self.dt
        
        # Apply periodic boundary conditions
        particle.position[0] = ((particle.position[0] + self.lx/2) % self.lx) - self.lx/2
        particle.position[1] = ((particle.position[1] + self.ly/2) % self.ly) - self.ly/2
    
    def dump_data(self, outfile):
        with open(outfile, 'w') as out:
            N = self.N
            lx = self.lx
            ly = self.ly
            out.write('{:.0f}\n'.format(N))
            out.write('{:.0f}   {:.0f}\n'.format(lx,ly))
            for p in self.particles:
                x, y = p.position
                theta = p.theta%(2*pi)
                out.write('{:.6f}  {:.6f}  {:.6f}\n'.format(x, y, theta))

    def simulate(self,t):
        for i in range(self.N):
            self.update_particle(i)
            if t%10==0:
                sim.dump_data(f'snapshot{t:05d}.txt')


sim = ParticleSimulation()
sim.initialise_file(outfile = 'raw.json')
duration = 1000
sim.read_init_config()
for t in range(duration):
    sim.simulate(t)
