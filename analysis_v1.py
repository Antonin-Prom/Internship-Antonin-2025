
import numpy as np
from msd_analysis import MSDAnalyzer
from trajs_builder import TrajBuilder
from Output_file_analyser_v1 import Output_analyzer
import matplotlib.pyplot as plt


def Energy_plot_all_compare(data1,data2):

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    time_line1,e_torque1, e_ela1, psy1 = data1
    time_line2,e_torque2, e_ela2, psy2 = data2

    total_energy1 = (e_torque1 + e_ela1)
    total_energy2 = (e_torque2 + e_ela2)

    axs[0].plot(total_energy1, color='blue',label = 'simu')
    axs[0].plot(total_energy2, color='cyan',label = 'tuto')
    axs[0].set_title('Total System Energy')
    axs[0].set_ylabel('Energy per Particle')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot(e_torque1 , color='red',label = 'simu')
    axs[1].plot(e_torque2 , color='orange',label = 'tuto')
    axs[1].set_title('Torque Energy')
    axs[1].set_ylabel('Energy per Particle')
    axs[1].grid(True)
    axs[1].legend()
    
    axs[2].plot(e_ela1, color='green',label = 'simu')
    axs[2].plot(e_ela2, color='lime',label = 'tuto')
    axs[2].set_title('Elastic Energy')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Energy per Particle')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_order_param(psi1,psi2):
    plt.plot(psi1,label = 'tuto')
    plt.plot(psi2,label = 'simu' )
    plt.title('Order parameter')
    plt.xlabel('Time Step')
    plt.ylabel('psi')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

e1 = Output_analyzer()
time_line1,e_torque1, e_ela1, psy1 = e1.read_snapshot(e1,filename = 'simul_snapout')
data1 = [time_line1,e_torque1, e_ela1, psy1]
e2 = Output_analyzer()
time_line2,e_torque2, e_ela2, psy2 = e2.read_snapshot(e2,filename = 'tuto_snapout')
data2 = [time_line2,e_torque2, e_ela2, psy2]
print(psy1,psy2)
Energy_plot_all_compare(data1,data2)
plot_order_param(psy1,psy2)