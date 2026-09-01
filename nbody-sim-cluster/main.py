# Runs the simulation for a stellar cluster and planetary system, saving the results and visualizations.

#Imports
import os
import shutil

from cluster_sim.constants import AU, M_sun, pc
from cluster_sim.generators import cluster
from cluster_sim.integrator import Hermite
from cluster_sim.save import save_simulation_data
from cluster_sim.visual import plot_trajectories

#Reset output directory function
def reset_output_directory(path):
	"""Remove previous output at path and recreate the directory."""
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path, exist_ok=True)

#Main simulation function
def main():
	# Configuration and initial parameters
	num_stars = 3
	cluster_radius = 0.05 * pc
	mass_mean = 0.0
	mass_std = 0.2
	virial_ratio_Q = 0.5               # Virial equilibrium: T / |U| = 0.5

	dt = 86400.0                       # Baseline timestep in seconds
	year = 31557600.0
	tmax = 100000.0 * year             # Total simulation time in seconds
	nu = 0.01                          # Timestep accuracy parameter
	n_order = 3                        # Predictor-corrector iterations
	save_snapshots = False             # Enable or disable snapshot saving
	snapshot_interval = 1000.0 * year  # Used when save_snapshots is True
	data_interval = 1000.0 * year        # Record diagnostics every x years
	progress_interval = 0.01 * tmax    # Report progress every 1% of the run
	output_dir = os.path.join("data", "processed", "test")
	snapshot_dir = os.path.join("data", "raw", "test")

	# Start with clean directories so old results cannot be mistaken for this run.
	reset_output_directory(output_dir)
	reset_output_directory(snapshot_dir)

	print("=" * 50)
	print("N-Body Solar System and Cluster Simulator")
	print("=" * 50)

	print("Initializing stellar cluster and planetary system...")
	x, y, z, vx, vy, vz, masses, is_star = cluster(
		num_stars, cluster_radius, mass_mean, mass_std, virial_ratio_Q
	)
	print(f"Initialized {len(masses)} total bodies.")
	star_indices = [index for index, star in enumerate(is_star) if star]
	host_index = min(star_indices, key=lambda index: abs(masses[index] - M_sun))

	print("Running Hermite integration...")
	results = Hermite(
		x, y, z, vx, vy, vz, masses,
		dt, tmax, nu, n_order, is_star,
		snapshot_interval if save_snapshots else None, data_interval,
		snapshot_dir, progress_interval
	)
	(
		xl, yl, zl, vxl, vyl, vzl,
		kel, gpel, tel, eccenl, angl, pl, al, ebl, timel
	) = results

	print("Saving simulation data...")
	save_simulation_data(
		output_dir,
		xl, yl, zl, vxl, vyl, vzl,
		kel, gpel, tel,
		eccenl, angl, pl, al, ebl,
		timel, masses
	)

	plot_trajectories(
		xl, yl, zl, len(masses),
		save_path=os.path.join(output_dir, "cluster_frame.png"),
		masses=masses,
		unit_scale=pc,
		unit_label="pc",
		title="N-Body Cluster Frame"
	)
	plot_trajectories(
		xl, yl, zl, len(masses),
		save_path=os.path.join(output_dir, "host_system.png"),
		reference_index=host_index,
		unit_scale=AU,
		unit_label="AU",
		title="Planetary System Relative to Host Star",
		body_indices=[host_index] + list(range(num_stars, len(masses)))
	)
	print("Simulation completed successfully.")


if __name__ == "__main__":
	main()

