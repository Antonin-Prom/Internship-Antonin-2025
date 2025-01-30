import numpy as np
from math import pi


class OutputGenerator:
    def __init__(self, lx=50, ly=50, a=1, k=1, J=1, snapshot_file = str,output_file = str,tmax = 1000, tjump = 10):
        self.lx = lx
        self.ly = ly
        self.a = a   
        self.k = k  
        self.J = J  
        self.rcut = 3 * a  
        
        self.snapshot_file = snapshot_file
        self.output_file = output_file
        self.positions = None
        self.thetas = None
        self.directions = None
        self.N = None  


        self.tmax = tmax
        self.tjump = tjump 
        self.time_line = np.arange(0,tmax,tjump)
        self.E_harm = 0
        self.E_torque = 0
        self.phi = 0 #param order
    @staticmethod
    def read_snapshot(filename: str):
        """Read particle positions and orientations from snapshot file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
            N = int(lines[0])  
            lx, ly = map(float, lines[1].split())  
            
            x = []
            y = []
            theta = []
            
            for line in lines[2:]:  
                xi, yi, theta_i = map(float, line.split())
                x.append(xi)
                y.append(yi)
                theta.append(theta_i)
                
        return N, lx, ly, np.array(x), np.array(y), np.array(theta)
    
    def load_snapshot(self, filename: str):
        """Load and process a snapshot file"""
        self.N, self.lx, self.ly, x, y, theta = self.read_snapshot(filename)
        self.positions = np.column_stack((x, y))
        self.thetas = theta
        self.directions = np.column_stack((np.cos(theta), np.sin(theta)))


    @staticmethod
    def elastic_energy(k, a, d):
        """Compute elastic energy between two particles"""
        return 0.5 * k * (2*a - d)**2
    
    @staticmethod
    def torque_energy(J, diff_angle):
        """Compute torque energy between two particles"""
        return -J * (np.cos(diff_angle) - 1)
    
    def compute_energies(self):
        """Compute all interaction energies for the system"""
        self.E_harm = 0
        self.E_torque = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                d = self.positions[i]- self.positions[j]
                d = np.linalg.norm(d)
                if d <= self.rcut:
                    if d < 2*self.a:
                        e_harm = self.elastic_energy(self.k, self.a, d)
                        self.E_harm += e_harm

                    diff_angle = self.thetas[j] - self.thetas[i]
                    e_torque = self.torque_energy(self.J, diff_angle)
                    self.E_torque += e_torque

    

    def order_param_directions(self):
        nx_sum = np.sum(self.directions[:,0])  
        ny_sum = np.sum(self.directions[:,1])
        return np.sqrt(nx_sum**2 + ny_sum**2) / self.N

    def order_param_theta(self):
        thetas = np.arctan2(self.directions[:,1], self.directions[:,0])  
        return np.abs(np.mean(np.exp(1j * thetas)))
        
    def output_at_t(self):

        with open(self.output_file, 'w') as out:
            for t in range(len(time_line)):
                filename = f'{self.snapshot_file}{time_line[t]:05d}.txt'
                self.load_snapshot(filename)
                self.compute_energies()
                self.phi = self.order_param_theta()
                out.write('{:.6f}  {:.6f}  {:.6f}   {:.6f}\n'.format(
                    time_line[t], self.E_torque/self.N, self.E_harm/self.N, self.phi))
                print(t)
                


                
                

d = OutputGenerator(snapshot_file = 'test_',output_file = 'tuto_snapout',tmax = 1000, tjump = 10)
tmax = 1000
t_jump = 10
time_line = np.arange(0,tmax,t_jump)
d.output_at_t()