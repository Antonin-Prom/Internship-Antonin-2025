# msd_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional, Union

class MSDAnalyzer:
    """
    Class for calculating and analyzing Mean Square Displacement (MSD) of particle trajectories.
    
    Attributes:
        trajs (np.ndarray): Particle trajectories
        tmax (int): Maximum time steps
        dt (float): Time step between frames
        tskip (int): Number of frames to skip between analysis points
    """
    
    def __init__(self, 
                 trajectories: np.ndarray,
                 dt: float = 0.1,
                 tskip: int = 10) -> None:
        """
        Initialize MSD analyzer with trajectories.
        
        Args:
            trajectories: Array of shape (n_particles, n_timesteps, 2) containing particle trajectories
            dt: Time step between frames
            tskip: Number of frames to skip between analysis points
        """
        self.trajs = np.array(trajectories)
        self.tmax = len(trajectories[0])
        self.dt = dt
        self.tskip = tskip
        
    @staticmethod
    def power_law(x: np.ndarray, a: float, alpha: float) -> np.ndarray:
        """Power law function of the form: y = a * x^alpha"""
        return a * x**alpha
    
    def calculate_msd(self, max_lag_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Mean Square Displacement.
        
        Args:
            max_lag_ratio: Maximum lag time as a fraction of total time (default: 0.25)
        
        Returns:
            Tuple of (lag_times, msd_values)
        """
        max_lag = int(self.tmax * max_lag_ratio)
        lag_times = np.arange(max_lag)
        msd_values = np.zeros(max_lag)
        
        for lag in lag_times:
            if lag == 0:
                continue
                
            displacements = []
            for traj in self.trajs:
                pos1 = traj[:-lag]
                pos2 = traj[lag:]
                diff = pos2 - pos1
                squared_displacement = np.sum(diff**2, axis=1)
                displacements.extend(squared_displacement)
            
            if displacements:
                msd_values[lag] = np.mean(displacements)
        
        # Convert to real time
        real_lag_times = lag_times * self.tskip * self.dt
        
        return real_lag_times, msd_values
    
    def fit_power_law(self, 
                     lag_times: np.ndarray, 
                     msd_values: np.ndarray,
                     fit_range: Optional[Tuple[int, int]] = None) -> Tuple[float, float, float, float]:
        """
        Fit power law to MSD data.
        
        Args:
            lag_times: Array of lag times
            msd_values: Array of MSD values
            fit_range: Tuple of (start_index, end_index) for fitting range
        
        Returns:
            Tuple of (a, alpha, a_err, alpha_err) where:
                a: amplitude
                alpha: power law exponent
                a_err: error in amplitude
                alpha_err: error in exponent
        """
        if fit_range is None:
            fit_start, fit_end = 1, len(lag_times)
        else:
            fit_start, fit_end = fit_range
            
        x_fit = lag_times[fit_start:fit_end]
        y_fit = msd_values[fit_start:fit_end]
        
        # Remove any zero or negative values
        mask = (y_fit > 0) & (x_fit > 0)
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]
        
        popt, pcov = curve_fit(self.power_law, x_fit, y_fit)
        a, alpha = popt
        a_err, alpha_err = np.sqrt(np.diag(pcov))
        
        return a, alpha, a_err, alpha_err
    
    def plot_msd(self, 
                 lag_times: np.ndarray, 
                 msd_values: np.ndarray,
                 fit_params: Optional[Tuple] = None,
                 title: str = 'Mean Square Displacement vs. Lag Time',
                 show_guides: bool = False) -> None:
        """
        Plot MSD data and optional power law fit.
        
        Args:
            lag_times: Array of lag times
            msd_values: Array of MSD values
            fit_params: Optional tuple of (a, alpha, a_err, alpha_err) from power law fit
            title: Plot title
            show_guides: Whether to show guide lines for t and t^2 scaling
        """
        plt.figure(figsize=(10, 6))
        
        # Plot MSD data
        plt.loglog(lag_times[1:], msd_values[1:], 'o', label='MSD data')
        
        # Plot fit if provided
        if fit_params is not None:
            a, alpha, a_err, alpha_err = fit_params
            x_smooth = np.logspace(np.log10(lag_times[1]), np.log10(lag_times[-1]), 100)
            plt.loglog(x_smooth, self.power_law(x_smooth, a, alpha), 'r-',
                      label=f'Fit: MSD ~ t^{alpha:.2f}±{alpha_err:.2f}')
            
            # Add fit parameters text box
            fit_text = f'α = {alpha:.2f} ± {alpha_err:.2f}\n'
            fit_text += f'A = {a:.2e} ± {a_err:.2e}'
            plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add guide lines if requested
        if show_guides:
            plt.loglog(lag_times[1:], 2*lag_times[1:], '--', color='black', label='~ t')
            plt.loglog(lag_times[1:], 2*lag_times[1:]**2, '--', color='red', label='~ t²')
        
        plt.xlabel('Lag Time')
        plt.ylabel('MSD')
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.show()
    
    def analyze(self, 
                fit_range: Optional[Tuple[int, int]] = None,
                title: str = 'Mean Square Displacement vs. Lag Time',
                show_guides: bool = False) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Perform complete MSD analysis: calculate MSD, fit power law, and plot results.
        
        Args:
            fit_range: Optional tuple of (start_index, end_index) for fitting range
            title: Plot title
            show_guides: Whether to show guide lines for t and t^2 scaling
        
        Returns:
            Tuple of (lag_times, msd_values, fit_params)
        """
        lag_times, msd_values = self.calculate_msd()
        fit_params = self.fit_power_law(lag_times, msd_values, fit_range)
        self.plot_msd(lag_times, msd_values, fit_params, title, show_guides)
        
        return lag_times, msd_values, fit_params