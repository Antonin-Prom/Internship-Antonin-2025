
import numpy as np
from msd_analysis import MSDAnalyzer
from trajs_builder import TrajBuilder
from OutputEnergyAnalyser import Output_analyzer
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

e = Output_analyzer(pattern = 'snapoutE*.txt')
e.rebuild()
e.Energy_plot()
