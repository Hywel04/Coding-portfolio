# Functions for visualizing simulation data.

#Imports
import os
import matplotlib.pyplot as plt
import numpy as np

#Plot trajectories of bodies in the simulation
def plot_trajectories(x, y, z, num_bodies, save_path="data/processed/orbital_path.png",
                      reference_index=None, masses=None, unit_scale=1.0,
                      unit_label="m", title="N-Body Orbital Paths", body_indices=None):
    """
    Renders and saves a 3D orbital trajectory plot for all bodies in the simulation.
    """
    if num_bodies < 1:
        raise ValueError("num_bodies must be at least 1")
    if unit_scale <= 0:
        raise ValueError("unit_scale must be positive")

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    z_arr = np.asarray(z)

    if x_arr.ndim != 2 or y_arr.ndim != 2 or z_arr.ndim != 2:
        raise ValueError("Trajectory arrays must have shape (timesteps, bodies)")
    if x_arr.shape != y_arr.shape or x_arr.shape != z_arr.shape:
        raise ValueError("Position trajectory arrays must have matching shapes")
    if x_arr.shape[1] < num_bodies:
        raise ValueError("num_bodies exceeds the number of trajectory columns")
    if reference_index is not None and not 0 <= reference_index < num_bodies:
        raise ValueError("reference_index must identify one of the plotted bodies")
    if body_indices is None:
        body_indices = list(range(num_bodies))
    else:
        body_indices = list(body_indices)
        if not body_indices or any(index < 0 or index >= num_bodies for index in body_indices):
            raise ValueError("body_indices must contain valid plotted body indices")

    if reference_index is not None:
        x_arr = x_arr - x_arr[:, [reference_index]]
        y_arr = y_arr - y_arr[:, [reference_index]]
        z_arr = z_arr - z_arr[:, [reference_index]]
    elif masses is not None:
        mass_arr = np.asarray(masses)
        if mass_arr.ndim != 1 or len(mass_arr) < num_bodies:
            raise ValueError("masses must contain one value for each plotted body")
        total_mass = np.sum(mass_arr[:num_bodies])
        if total_mass <= 0:
            raise ValueError("masses must have a positive total")
        x_arr = x_arr - np.sum(x_arr * mass_arr[:num_bodies], axis=1, keepdims=True) / total_mass
        y_arr = y_arr - np.sum(y_arr * mass_arr[:num_bodies], axis=1, keepdims=True) / total_mass
        z_arr = z_arr - np.sum(z_arr * mass_arr[:num_bodies], axis=1, keepdims=True) / total_mass

    x_arr = x_arr / unit_scale
    y_arr = y_arr / unit_scale
    z_arr = z_arr / unit_scale

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(title)
    ax.set_xlabel(f'X ({unit_label})')
    ax.set_ylabel(f'Y ({unit_label})')
    ax.set_zlabel(f'Z ({unit_label})')

    for i in body_indices:
        ax.plot(x_arr[:, i], y_arr[:, i], z_arr[:, i], label=f'Body {i}')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    output_directory = os.path.dirname(save_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✓ Trajectory plot saved to: {save_path}")