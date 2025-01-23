#Output file analyser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

class Output_analyzer:

    def __init__(self,pattern: str = 'snapoutE*.txt'):
        self.pattern = pattern
        self.Ela = None
        self.Etorque = None
        self.nx = None
        self.ny = None
        self.total_time_data = None
        
        
    @staticmethod
    def read_snapshot(filename: str) :
        with open(filename, 'r') as f:
            N = N = len(f.readlines())
            e_torque = []
            e_ela = []
            nx = []
            ny = []

            for _ in range(N):
                _, e_tor,e_el,px,py = map(float, f.readline().split())
                e_torque.append(e_tor)
                e_ela.append(e_el)
                nx.append(px)
                ny.append(py)
                
        return np.array(e_torque), np.array(e_ela),np.array(nx),np.array(ny) 
    
    def rebuild(self):
        files = sorted(glob.glob(self.pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {self.pattern}")
        q, *_ = self.read_snapshot(files[0])
        N = len(q)
        ite_max = len(files)
        e_torque_all_time = np.zeros((ite_max, N))
        e_ela_all_time = np.zeros((ite_max, N))
        nx_all_time = np.zeros((ite_max, N))
        ny_all_time = np.zeros((ite_max, N))
        
        for ite, filename in enumerate(files):
            e_torque, e_ela, nx, ny = self.read_snapshot(filename)
            e_torque_all_time[ite] = e_torque
            e_ela_all_time[ite] = e_ela
            nx_all_time[ite] = nx
            ny_all_time[ite] = ny
        
        self.total_time_data = {
            'Etorque': e_torque_all_time,
            'Ela': e_ela_all_time,
            'nx': nx_all_time,
            'ny': ny_all_time
        }

    def Energy_plot(self, energy_type='total'):
            plt.figure(figsize=(10, 6))
            
            if energy_type == 'total':
                total_energy = (self.total_time_data['Etorque'] + 
                                self.total_time_data['Ela']).sum(axis=1)
                plt.plot(total_energy, label='Total Energy')
                plt.title('Total System Energy over Time')
            
            elif energy_type == 'torque':
                torque_energy = self.total_time_data['Etorque'].sum(axis=1)
                plt.plot(torque_energy, label='Torque Energy', color='red')
                plt.title('Torque Energy over Time')
            
            elif energy_type == 'elastic':
                elastic_energy = self.total_time_data['Ela'].sum(axis=1)
                plt.plot(elastic_energy, label='Elastic Energy', color='green')
                plt.title('Elastic Energy over Time')
            
            plt.xlabel('Time Step')
            plt.ylabel('Energy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
