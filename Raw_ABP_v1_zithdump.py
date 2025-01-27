#Brute_force
import numpy as np
from math import sin, cos, sqrt, pi
from random import uniform
from random import gauss
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import vtk
from msd_analysis import MSDAnalyzer
from trajs_builder import TrajBuilder

class Particle:
    def __init__(self, id, position, theta, velocity=[0.0, 0.0], force=[0.0, 0.0]):
        self.id = id
        self.position = np.array(position)
        self.theta = theta
        self.direction = np.array([cos(theta), sin(theta)])
        self.d_theta = 0
        self.displacement = np.array(velocity)
        self.force = np.array(force)
        self.tau = 0


class ParticleSimulation:
    def __init__(self, lx=20, ly=20, density=0.4, a=1, v0=0.1, noise_amplitude=0.01,
                 k=1, J=1, gamma_t=1, gamma_rot=1, Dr=1, Dt= 0, dt=0.1):
        
        #Box parameters
        self.lx, self.ly = lx, ly
        self.xmin, self.xmax = 0, lx
        self.ymin, self.ymax = 0, ly

        # Particle parameters
        self.N = int(lx*ly*density)
        self.a = a  
        self.rcut = 10*a
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
        self.dump_jump = 2 #dump every 10dt
        self.particles_positions = None
        self.particles_displacement = np.zeros((self.N,2),dtype=float)
        self.particles_d_theta =np.zeros(self.N)
        self.particles_thetas = None

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
            self.particles_positions = np.array([p.position for p in self.particles])
            self.particles_thetas = np.array([p.theta for p in self.particles])
    def harmonic_force(self, d):
        return -self.k * (self.a - d)     

    def calculate_force_torque(self, i):
        particle = self.particles[i]
        force = np.zeros(2)
        torque = 0
        self.E_ela,self.E_torque = 0,0
        xi = particle.position[0]%self.lx
        yi = particle.position[1]%self.ly
        for j in range(i+1,self.N):   
            neighbor = self.particles[j]
            xj = neighbor.position[0]%self.lx
            yj = neighbor.position[1]%self.ly
            diff = particle.position - neighbor.position
            diff[0] = diff[0] - self.lx * round(diff[0] / self.lx)
            diff[1] = diff[1] - self.ly * round(diff[1] / self.ly)
            d = np.linalg.norm(diff)
            if d <= self.rcut:
                if d < 2*self.a:
                    force_magnitude = self.harmonic_force(d)
                    particle.force += force_magnitude * (diff / d)
                    neighbor.force -= force_magnitude * (diff / d)
                tau_z = self.J*(xi*yj - yi*xj)
                particle.tau += tau_z 
                neighbor.tau -= tau_z
                
    
    def update_displacements(self,i):     
        particle = self.particles[i]
        self.calculate_force_torque(i)
        #thermal_noise_rot = sqrt(2*self.Dr*self.dt)*gauss(0,1)
        thermal_noise_rot = uniform(-self.noise_amplitude,+self.noise_amplitude)
        thermal_amplitude_trans = sqrt(2*self.Dt*self.dt)

        noise_vector = np.array([gauss(0,1), gauss(0,1)]) * thermal_amplitude_trans
        particle.displacement = (self.v0 * particle.direction + particle.force/self.gamma_t) * self.dt + noise_vector
        particle.d_theta += (particle.tau/self.gamma_rot) * self.dt + thermal_noise_rot
        self.particles_displacement[i] = particle.displacement
        self.particles_d_theta[i] = particle.d_theta


    def dump_data(self, outfile):
        with open(outfile, 'w') as out:
            N = self.N
            lx = self.lx
            ly = self.ly
            out.write('{:.0f}\n'.format(N))
            out.write('{:.0f} {:.0f}\n'.format(lx,ly))
            for i in range(self.N):
                x, y = self.particles_positions[i]
                theta = self.particles_thetas[i] % (2*np.pi)
                out.write('{:.6f} {:.6f} {:.6f}\n'.format(x, y, theta))

    def simulate(self,t):
        for i in range(self.N):
            self.update_displacements(i)

        self.particles_positions += self.particles_displacement
        
        self.particles_positions[:, 0] %= self.lx
        self.particles_positions[:, 1] %= self.ly
        
        self.particles_thetas += self.particles_d_theta
        
        if t % self.dump_jump == 0:
            self.dump_data(f'10snapshot{t:05d}.txt')


sim = ParticleSimulation()
sim.initialise_file(outfile = 'raw.json')
duration = 500
sim.read_init_config()
for t in range(duration):
    sim.simulate(t)
