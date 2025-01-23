import numpy as np
import glob
from typing import Tuple, List, Optional

class TrajBuilder:
    """
    Class for building particle trajectories from snapshot files.
    
    Attributes:
        pattern (str): Glob pattern for snapshot files
        trajs (np.ndarray): Array of particle trajectories once built
    """
    
    def __init__(self, pattern: str = 'snapshot*.txt') -> None:
        self.pattern = pattern
        self.trajs = None
        self.angles = None
        
    @staticmethod
    def read_snapshot(filename: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
        with open(filename, 'r') as f:
            N = int(f.readline())
            lx, ly = map(float, f.readline().split())
            positions = []
            thetas = []
            for _ in range(N):
                x, y, theta = map(float, f.readline().split())
                positions.append([x, y])
                thetas.append(theta)
                
        return np.array(positions), np.array(thetas), lx, ly
    
    def build_trajectories(self) -> np.ndarray:

        files = sorted(glob.glob(self.pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {self.pattern}")     
        positions_all_time = []
        tmax = len(files)
        first_positions, _, _, _ = self.read_snapshot(files[0])
        N = len(first_positions)
    
        for filename in files:
            positions, _, _, _ = self.read_snapshot(filename)
            positions_all_time.append(positions)
            
        self.trajs = np.array([
            [[positions_all_time[t][i][0], positions_all_time[t][i][1]]
             for t in range(tmax)]
            for i in range(N)
        ])
        
        return self.trajs
    
    def get_angles(self):

        files = sorted(glob.glob(self.pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {self.pattern}")     
        angle_all_time = []
        tmax = len(files)
        _, angle, _, _ = self.read_snapshot(files[0])
        N = len(angle)
    
        for filename in files:
            _, angle, _, _  = self.read_snapshot(filename)
            angle_all_time.append(angle)
            
        self.angles = np.array([
            [angle_all_time[t][i] for t in range(tmax)]
            for i in range(N)])
        
        return self.angles
        

    def get_trajectories(self) -> Optional[np.ndarray]:
        if self.trajs is None:
            try:
                return self.build_trajectories()
            except Exception as e:
                print(f"Error building trajectories: {e}")
                return None
        return self.trajs