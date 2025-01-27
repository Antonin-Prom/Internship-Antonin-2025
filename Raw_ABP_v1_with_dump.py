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



class ParticleSimulation:
    def __init__(self, lx=50, ly=50, density=0.4, a=1, v0=1,
                 k=10, J=1, gamma_t=1, gamma_rot=1, T=0.1, dt=0.1):
        
        #Box parameters
        self.lx, self.ly = lx, ly
        self.xmin, self.xmax = 0, lx
        self.ymin, self.ymax = 0, ly

        # Particle parameters
        self.N = int(lx*ly*density)
        self.a = a  
        self.rcut = 3*a
        self.v0 = v0
        
        # Physical parameters
        self.k = k
        self.J = J
        self.gamma_t = gamma_t
        self.gamma_rot = gamma_rot
        self.T = T
        self.dt = dt
        self.Dr = self.T/self.gamma_rot
        self.Dt = self.T/self.gamma_t

        #Initalise
        self.particles = []
        self.dump_jump = 10 #dump every 10dt
        self.particles_positions = None
        self.particles_displacement = np.zeros((self.N,2),dtype=float)
        self.particles_d_theta =np.zeros(self.N)
        self.particles_F_harm= np.zeros((self.N,2),dtype=float)
        self.particles_Tau = np.zeros(self.N)
        self.particles_E_harm = np.zeros(self.N)
        self.particles_E_torque = np.zeros(self.N)
        self.particles_thetas = None
        self.particles_direction = np.zeros((self.N,2),dtype=float)

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
            self.particles_direction = np.array([[cos(p.theta),sin(p.theta)] for p in self.particles])

    def harmonic_force(self, d):
        return self.k * (2*self.a - d)     

    def elastic_energy(self, d):
        return 0.5*self.k * (2*self.a - d)**2  

    def torque_energy(self,diff_angle):
        return -self.J*(cos(diff_angle) - 1)
    
    def calculate_force_torque(self, i):
        
        for j in range(i+1,self.N):   

            diff = self.particles_positions[i]- self.particles_positions[j]
            diff[0] = diff[0] - self.lx * round(diff[0] / self.lx)
            diff[1] = diff[1] - self.ly * round(diff[1] / self.ly)
            d = np.linalg.norm(diff)
            if d <= self.rcut:
                if d < 2*self.a:
                    force_magnitude = self.harmonic_force(d)
                    self.particles_F_harm[i] += force_magnitude * (diff / d)
                    self.particles_F_harm[j] -= force_magnitude * (diff / d)
                    self.particles_E_harm[i] += self.elastic_energy(d)/2
                    self.particles_E_harm[j] += self.elastic_energy(d)/2

                diff_angle = self.particles_thetas[j] - self.particles_thetas[i]
                tau_z = self.J*sin(diff_angle)
                self.particles_Tau[i] += tau_z 
                self.particles_Tau[j] -= tau_z

                self.particles_E_torque[i] += self.torque_energy(diff_angle)/2
                self.particles_E_torque[j] += self.torque_energy(diff_angle)/2            

    def update_displacements(self,i):     
        particle = self.particles[i]
        self.calculate_force_torque(i)
        thermal_noise_rot = sqrt(2*self.Dr*self.dt)*gauss(0,1)
        thermal_amplitude_trans = sqrt(2*self.Dt*self.dt)

        noise_vector = np.array([gauss(0,1), gauss(0,1)]) * thermal_amplitude_trans
        self.particles_displacement[i]  = (self.v0 * self.particles_direction[i] + self.particles_F_harm[i]/self.gamma_t) * self.dt + noise_vector
        self.particles_d_theta[i] = (self.particles_Tau[i]/self.gamma_rot) * self.dt + thermal_noise_rot


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

    def dump_analysis(self, outfile, t):
        with open(outfile, 'w') as out:
            for i in range(self.N):
                out.write('{:.6f}  {:.6f}  {:.6f}   {:.6f}  {:.6f}\n'.format(
                    t, self.particles_E_torque[i],self.particles_E_harm[i],self.particles_direction[i][0],self.particles_direction[i][1]))

    def simulate(self,t):
        #reset
        self.particles_F_harm= np.zeros((self.N,2),dtype=float)
        self.particles_Tau = np.zeros(self.N)
        self.particles_E_harm = np.zeros(self.N)
        self.particles_E_torque = np.zeros(self.N) 

        for i in range(self.N):
            self.update_displacements(i)

        if t % self.dump_jump == 0:
            self.dump_data(f'1000snapshot{t:05d}.txt')
            self.dump_analysis( f'1000snapoutE{t:05d}.txt', t)

        self.particles_positions += self.particles_displacement
        
        self.particles_positions[:, 0] %= self.lx
        self.particles_positions[:, 1] %= self.ly
        
        self.particles_thetas += self.particles_d_theta
        self.particles_direction = np.array([[cos(self.particles_thetas[i]),sin(self.particles_thetas[i])] for i in range(self.N)])

        



sim = ParticleSimulation()
sim.initialise_file(outfile = 'raw.json')
duration = 1000
sim.read_init_config()
for t in range(duration):
    sim.simulate(t)
