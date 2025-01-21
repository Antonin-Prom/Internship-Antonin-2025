import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob



def read_snapshot(filename):
    """Read a single snapshot file and return particle data"""
    with open(filename, 'r') as f:
        # Read number of particles
        N = int(f.readline())
        # Read box dimensions
        lx, ly = map(float, f.readline().split())
        
        # Read particle positions and orientations
        positions = []
        thetas = []
        for _ in range(N):
            x, y, theta = map(float, f.readline().split())
            positions.append([x, y])
            thetas.append(theta)
            
    return np.array(positions), np.array(thetas), lx, ly

def animate_snapshots(snapshot_pattern='snapshot*.txt'):

    # Get list of snapshot files sorted by number
    files = sorted(glob.glob(snapshot_pattern))
    positions_all_time = []
    incr = 10
    tmax = len(files)
    N = len(read_snapshot(files[0])[0])
    print(N)
    for t in range(tmax):
        positions_all_time.append(read_snapshot(files[t]))
    print(positions_all_time[0][0][0]) #[t][0 = pos, 1 = theta][num particule]
    trajs = []
    for i in range(N):
        trajs.append(positions_all_time[:][0][i])
    '''
    lag_times = np.arange(tmax)
    for traj in trajs:
    for i in range(tmax):
        lag = lag_times[i]
        msd = np.mean((positions_all_time),axis = 0)'''


animate_snapshots()