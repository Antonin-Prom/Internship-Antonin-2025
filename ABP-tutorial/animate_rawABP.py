import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

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
    if not files:
        raise ValueError(f"No files found matching pattern: {snapshot_pattern}")
    
    # Read first snapshot to get dimensions
    positions, thetas, lx, ly = read_snapshot(files[0])
    N = len(positions)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-lx/2, lx/2)
    ax.set_ylim(-ly/2, ly/2)
    ax.set_title('Particle Animation')
    
    # Create scatter plot for positions
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=30, c='blue')
    
    # Create quiver plot for orientations
    directions = np.column_stack([np.cos(thetas), np.sin(thetas)])
    quiver = ax.quiver(positions[:, 0], positions[:, 1],
                      directions[:, 0], directions[:, 1],
                      scale=25)
    
    def update(frame):
        # Read new snapshot
        filename = files[frame]
        positions, thetas, _, _ = read_snapshot(filename)
        
        # Update scatter positions
        scatter.set_offsets(positions)
        
        # Update quiver
        directions = np.column_stack([np.cos(thetas), np.sin(thetas)])
        quiver.set_offsets(positions)
        quiver.set_UVC(directions[:, 0], directions[:, 1])
        
        # Update title with frame number
        ax.set_title(f'Frame {frame}')
        
        return scatter, quiver
    
    # Create animation
    num_frames = len(files)
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                interval=50, blit=True)
    
    plt.show()
    return ani

if __name__ == "__main__":
    # Animate all snapshot files in current directory
    animate_snapshots()