
import numpy as np
from msd_analysis import MSDAnalyzer
from trajs_builder import TrajBuilder
from OutputEnergyAnalyser import Output_analyzer
import matplotlib.pyplot as plt
# Create or load your trajectories

def traj_related():
    builder = TrajBuilder(pattern="snapshot*.txt")
    trajectories = builder.get_trajectories()
    print(len(trajectories))
    analyzer = MSDAnalyzer(trajectories, dt=0.1, tskip=10)

    lag_times, msd_values, fit_params = analyzer.analyze(title="MSD Analysis",show_guides=True)

    lag_times, msd_values = analyzer.calculate_msd(max_lag_ratio=0.25)
    fit_params = analyzer.fit_power_law(lag_times, msd_values, fit_range=(1, 100))
    analyzer.plot_msd(lag_times, msd_values, fit_params, 
                    title="Custom Analysis", 
                    show_guides=True)
    

def Energy_plot_all_compare(data1,data2):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    

    total_energy1 = (data1['Etorque'] + 
                    data1['Ela']).sum(axis=1)
    torque_energy1 = data1['Etorque'].sum(axis=1)
    elastic_energy1 = data1['Ela'].sum(axis=1)

    total_energy2 = (data2['Etorque'] + 
                data2['Ela']).sum(axis=1)
    torque_energy2 = data2['Etorque'].sum(axis=1)
    elastic_energy2 = data2['Ela'].sum(axis=1)
    
    axs[0].plot(total_energy1, color='navy',label = 'tuto')
    axs[0].plot(total_energy2, color='royalblue',label = 'simu')
    axs[0].set_title('Total System Energy')
    axs[0].set_ylabel('Energy per Particle')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(torque_energy1, color='red',label = 'tuto')
    axs[1].plot(torque_energy2, color='darkred',label = 'simu')
    axs[1].set_title('Torque Energy')
    axs[1].set_ylabel('Energy per Particle')
    axs[1].grid(True)
    axs[1].legend()
    
    axs[2].plot(elastic_energy1, color='lime',label = 'tuto')
    axs[2].plot(elastic_energy2, color='darkgreen',label = 'simu' )
    axs[2].set_title('Harmonic potential')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Energy per Particle')
    axs[2].grid(True)
    axs[2].legend()
    plt.tight_layout()
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

e1 = Output_analyzer(pattern = 'simu_snapout*.txt')
e2 = Output_analyzer(pattern = 'test_snapout*.txt')
data1 = e1.rebuild()
data2 = e2.rebuild()
Energy_plot_all_compare(data1,data2)
psi1 = e1.order_param_theta()
psi2 = e2.order_param_theta()
plot_order_param(psi1,psi2)