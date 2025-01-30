#Output file analyser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
from math import sin, cos, sqrt, pi

class Output_analyzer:

    def __init__(self):
        self.time_line = None
        self.e_torque= None
        self.e_ela = None
        self.psy = None
        
    @staticmethod
    def read_snapshot(self, filename: str) :
        with open(filename, 'r') as f:
            lines = f.readlines()

            time_line = np.zeros(len(lines))
            e_torque = np.zeros(len(lines))
            e_ela = np.zeros(len(lines))
            psy = np.zeros(len(lines))
            
            for t in range(len(lines)):
                time_line[t], e_torque[t], e_ela[t] , psy[t] = map(float, lines[t].split())

            self.time_line = time_line
            self.e_torque= e_torque
            self.e_ela = e_ela
            self.psy = psy        

        return time_line,e_torque, e_ela, psy
    

    def Energy_plot_all(self, energy_type='all'):
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        total_energy = (self.e_torque + self.e_ela)
        
        axs[0].plot(total_energy, color='blue')
        axs[0].set_title('Total System Energy')
        axs[0].set_ylabel('Energy per Particle')
        axs[0].grid(True)
        
        axs[1].plot(self.e_torque , color='red')
        axs[1].set_title('Torque Energy')
        axs[1].set_ylabel('Energy per Particle')
        axs[1].grid(True)
        
        axs[2].plot(self.e_ela, color='green')
        axs[2].set_title('Elastic Energy')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Energy per Particle')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
 

analy = Output_analyzer()



