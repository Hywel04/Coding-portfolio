# Functions for saving simulation data to text and CSV files.

#Imports
import os
import numpy as np


#Save functions
def safe_str(value):
	"""Convert a value to a string, using ``NaN`` for invalid values."""
	try:
		if isinstance(value, (int, float)):
			return str(value)
		return str(float(value))
	except (ValueError, TypeError):
		return "NaN"


def save_snapshot(snapshot_counter, time_list, x_pos_list, y_pos_list, z_pos_list,
				  x_vel_list, y_vel_list, z_vel_list, KE_list, GPE_list, TE_list,
				  eccentricity_list, angular_momentum_list, period_list, num_bodies,
				  output_folder="."):
	"""Save simulation output for each recorded time step to a text file."""
	os.makedirs(output_folder, exist_ok=True)
	snapshot_filename = os.path.join(output_folder, f"snapshot_{snapshot_counter}.txt")
	num_timesteps = len(time_list)

	x_pos_list = np.array(x_pos_list).T.tolist()
	y_pos_list = np.array(y_pos_list).T.tolist()
	z_pos_list = np.array(z_pos_list).T.tolist()
	x_vel_list = np.array(x_vel_list).T.tolist()
	y_vel_list = np.array(y_vel_list).T.tolist()
	z_vel_list = np.array(z_vel_list).T.tolist()

	assert len(KE_list) == len(GPE_list) == len(TE_list) == num_timesteps, \
		"Energy lists must match number of time steps"
	assert len(eccentricity_list) == len(angular_momentum_list) == len(period_list) == num_timesteps, \
		"Orbital lists must match number of time steps"

	header = "Time"
	for body_index in range(num_bodies):
		header += " {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
			f"B{body_index}_X", f"B{body_index}_Y", f"B{body_index}_Z",
			f"B{body_index}_Vx", f"B{body_index}_Vy", f"B{body_index}_Vz"
		)
	header += " KE        GPE       TE"

	orbital_count = max((len(values) for values in eccentricity_list), default=0)
	for orbital_index in range(orbital_count):
		header += " {:<15} {:<15} {:<15}".format(
			f"Ecc_{orbital_index}", f"AngMom_{orbital_index}", f"Period_{orbital_index}"
		)

	with open(snapshot_filename, "w") as file:
		file.write(header + "\n")

		for time_index in range(num_timesteps):
			row_data = [time_list[time_index]]
			for body_index in range(num_bodies):
				row_data.extend([
					x_pos_list[body_index][time_index],
					y_pos_list[body_index][time_index],
					z_pos_list[body_index][time_index],
					x_vel_list[body_index][time_index],
					y_vel_list[body_index][time_index],
					z_vel_list[body_index][time_index],
				])

			row_data.extend([
				KE_list[time_index],
				GPE_list[time_index],
				TE_list[time_index],
			])

			for orbital_index in range(orbital_count):
				row_data.extend([
					eccentricity_list[time_index][orbital_index]
					if orbital_index < len(eccentricity_list[time_index]) else np.nan,
					angular_momentum_list[time_index][orbital_index]
					if orbital_index < len(angular_momentum_list[time_index]) else np.nan,
					period_list[time_index][orbital_index]
					if orbital_index < len(period_list[time_index]) else np.nan,
				])

			file.write(" ".join(f"{value:<10.4f}" for value in row_data) + "\n")


def save_array(filename, data):
	"""Save data as a comma-separated text array."""
	try:
		array = np.asarray(data)
	except ValueError:
		# Orbital histories can have missing entries for unbound bodies.
		rows = list(data)
		width = max((len(row) for row in rows), default=0)
		array = np.full((len(rows), width), np.nan)
		for row_index, row in enumerate(rows):
			array[row_index, :len(row)] = row

	np.savetxt(filename, array, delimiter=",", fmt="%.6f")


def save_simulation_data(folder, xl, yl, zl, vxl, vyl, vzl, kel, gpel, tel,
						 eccenl, angl, pl, al, ebl, timel, mass):
	"""Save simulation data as separate CSV files in ``folder``."""
	os.makedirs(folder, exist_ok=True)

	save_array(os.path.join(folder, "timel.csv"), [timel])
	save_array(os.path.join(folder, "kel.csv"), [kel])
	save_array(os.path.join(folder, "gpel.csv"), [gpel])
	save_array(os.path.join(folder, "tel.csv"), [tel])
	save_array(os.path.join(folder, "mass.csv"), [mass])

	save_array(os.path.join(folder, "xl.csv"), xl)
	save_array(os.path.join(folder, "yl.csv"), yl)
	save_array(os.path.join(folder, "zl.csv"), zl)
	save_array(os.path.join(folder, "vxl.csv"), vxl)
	save_array(os.path.join(folder, "vyl.csv"), vyl)
	save_array(os.path.join(folder, "vzl.csv"), vzl)

	save_array(os.path.join(folder, "eccenl.csv"), eccenl)
	save_array(os.path.join(folder, "angl.csv"), angl)
	save_array(os.path.join(folder, "pl.csv"), pl)
	save_array(os.path.join(folder, "al.csv"), al)
	save_array(os.path.join(folder, "ebl.csv"), ebl)

	print(f"Data saved to folder: {folder}")
