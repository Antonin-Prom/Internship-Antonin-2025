#Output file analyser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
from math import sin, cos, sqrt, pi

class Output_analyzer:

    def __init__(self,pattern: str = 'snapoutE*.txt'):
        self.pattern = pattern
        self.Ela = None
        self.Etorque = None
        self.nx = None
        self.ny = None
        self.total_time_data = None
        self.N = None
        
    @staticmethod
    def read_snapshot(filename: str) :
        with open(filename, 'r') as f:
            lines = f.readlines()
            e_torque = []
            e_ela = []
            nx = []
            ny = []

            for line in lines:
                t, e_tor, e_el, px, py = map(float, line.split())
                e_torque.append(e_tor)
                e_ela.append(e_el)
                nx.append(px)
                ny.append(py)
                
        return np.array(e_torque), np.array(e_ela),np.array(nx),np.array(ny),np.array(t), 
    
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
            t,e_torque, e_ela, nx, ny = self.read_snapshot(filename)
            e_torque_all_time[ite] = e_torque/N
            e_ela_all_time[ite] = e_ela/N
            nx_all_time[ite] = nx
            ny_all_time[ite] = ny
        
        self.total_time_data = {
            'time' : t,
            'Etorque': e_torque_all_time,
            'Ela': e_ela_all_time,
            'nx': nx_all_time,
            'ny': ny_all_time
        }
        self.N = N

        return self.total_time_data

    def order_param_theta(self):

        self.nx = self.total_time_data['nx']
        self.ny = self.total_time_data['ny']
        thetas = np.arctan2(self.ny, self.nx)

        psi = np.mean(np.exp(1j * thetas),axis=1)
        
        return np.abs(psi)

    def order_param_directions(self):

        self.nx = self.total_time_data['nx']
        self.ny = self.total_time_data['ny']
        nx_sum = np.sum(self.nx)
        ny_sum = np.sum(self.ny)
        
        # Calculate magnitude of the total normalized velocity
        phi = np.sqrt(nx_sum**2 + ny_sum**2) / (self.N)
        
        return phi
    
    def plot_order_param(self):
        psi = self.order_param_directions()
        plt.plot(psi)
        plt.title('Order parameter')
        plt.xlabel('Time Step')
        plt.ylabel('psi')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def Energy_plot(self, energy_type='total'):
            plt.figure(figsize=(10, 6))
            
            if energy_type == 'total':
                total_energy = (self.total_time_data['Etorque'] + 
                                self.total_time_data['Ela']).sum(axis=1)/self.N

                plt.plot(total_energy, label='Total Energy')
                plt.title('Total System Energy over Time')
            
            elif energy_type == 'torque':
                torque_energy = self.total_time_data['Etorque'].sum(axis=1)/self.N
                plt.plot(torque_energy, label='Torque Energy', color='red')
                plt.title('Torque Energy over Time')
            
            elif energy_type == 'elastic':
                elastic_energy = self.total_time_data['Ela'].sum(axis=1)/self.N
                plt.plot(elastic_energy, label='Elastic Energy', color='green')
                plt.title('Elastic Energy over Time')
            
            plt.xlabel('Time Step')
            plt.ylabel('Energy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def Energy_plot_all(self, energy_type='all'):
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        total_energy = (self.total_time_data['Etorque'] + 
                        self.total_time_data['Ela']).sum(axis=1)
        torque_energy = self.total_time_data['Etorque'].sum(axis=1)
        elastic_energy = self.total_time_data['Ela'].sum(axis=1)
        
        axs[0].plot(total_energy/self.N, color='blue')
        axs[0].set_title('Total System Energy')
        axs[0].set_ylabel('Energy per Particle')
        axs[0].grid(True)
        
        axs[1].plot(torque_energy/self.N, color='red')
        axs[1].set_title('Torque Energy')
        axs[1].set_ylabel('Energy per Particle')
        axs[1].grid(True)
        
        axs[2].plot(elastic_energy/self.N, color='green')
        axs[2].set_title('Elastic Energy')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Energy per Particle')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
