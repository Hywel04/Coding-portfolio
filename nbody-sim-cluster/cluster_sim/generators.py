#Initial conditions for the stellar cluster and planet generators

#Imports
import numpy as np
from .constants import G, AU, pc, M_sun, epsilon

#Generate stars
def generate_stars(N, R, mass_mean, mass_std, Q):
    if Q <= 0:
        raise ValueError("Q must be positive")

    # Sample star masses from a log-normal distribution.
    masses = np.random.lognormal(
        mean=mass_mean, sigma=mass_std, size=N - 1
    ) * M_sun
    masses = np.append(masses, M_sun)

    x_positions, y_positions, z_positions = [], [], []
    is_star = np.ones(N, dtype=bool)

    # Generate positions using a Gaussian and filter by cluster radius R.
    while len(x_positions) < N:
        x, y, z = np.random.normal(0, R / 3, size=3)
        if np.sqrt(x**2 + y**2 + z**2) <= R:
            x_positions.append(x)
            y_positions.append(y)
            z_positions.append(z)

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    z_positions = np.array(z_positions)

    M_tot = np.sum(masses)
    U = -(3 / 5) * G * M_tot**2 / R
    T = Q * np.abs(U)

    # Generate random velocities, remove bulk motion, and normalize their energy.
    vx, vy, vz = [], [], []
    while len(vx) < N:
        v_x, v_y, v_z = np.random.uniform(0, 1, size=3)
        vx.append(v_x)
        vy.append(v_y)
        vz.append(v_z)

    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    total_mass = np.sum(masses)
    vx -= np.sum(masses * vx) / total_mass
    vy -= np.sum(masses * vy) / total_mass
    vz -= np.sum(masses * vz) / total_mass

    current_kinetic = 0.5 * np.sum(masses * (vx**2 + vy**2 + vz**2))
    velocity_scale = np.sqrt(T / current_kinetic)
    vx *= velocity_scale
    vy *= velocity_scale
    vz *= velocity_scale

    return x_positions, y_positions, z_positions, vx, vy, vz, masses, is_star

#Generate/add solar system planets around a star
def planet(x, y, z, vx, vy, vz, masses, is_star, id):
    # Select the star around which to generate planets
    star_x, star_y, star_z = x[id], y[id], z[id]
    star_vx, star_vy, star_vz = vx[id], vy[id], vz[id]
    star_mass = masses[id]

    # Planetary semi-major axes (AU) and masses (kg) based on Solar System values
    solar_system = {
        "Mercury": (0.387, 3.302e23),
        "Venus": (0.723, 4.8685e24),
        "Earth": (1.00, 5.972e24),
        "Mars": (1.52, 0.64169e24),
        "Jupiter": (5.20, 1898.6e24),
        "Saturn": (9.58, 568.34e24),
        "Uranus": (19.2, 86.813e24),
        "Neptune": (30.2, 102.413e24)
    }

    planets = []

    # Set the inclination of the planets' orbits, either set to a fixed value or randomize it
    #phi = np.random.uniform(0, np.pi)  # Inclination
    phi = 0 #Solar system value

    for name, (a, p_mass) in solar_system.items():
        # Convert semi-major axis from AU to meters
        a *= AU

        # Random orbital phase (0 to 2π)
        phase = np.random.uniform(0, 2 * np.pi)

        # Position in the orbital plane
        xp = a * np.cos(phase)
        yp = a * np.sin(phase)
        zp = 0

        # Circular orbit velocity magnitude
        v_orbit = np.sqrt(
            G * star_mass * a**2 / (a**2 + epsilon**2)**1.5
        )
        vxp = -v_orbit * np.sin(phase)
        vyp = v_orbit * np.cos(phase)
        vzp = 0

        # Rotation matrix for inclination around the x-axis
        r = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        # Apply rotation
        p_vector = np.array([xp, yp, zp])
        v_vector = np.array([vxp, vyp, vzp])

        p_rot = r @ p_vector
        v_rot = r @ v_vector

        # Adjust for host star position and velocity
        xp, yp, zp = star_x + p_rot[0], star_y + p_rot[1], star_z + p_rot[2]
        vxp, vyp, vzp = star_vx + v_rot[0], star_vy + v_rot[1], star_vz + v_rot[2]

        # Append planet properties and mark as a planet
        planets.append((name, xp, yp, zp, vxp, vyp, vzp, p_mass))

        # Append the planet data to the respective arrays
        x = np.append(x, xp)
        y = np.append(y, yp)
        z = np.append(z, zp)
        vx = np.append(vx, vxp)
        vy = np.append(vy, vyp)
        vz = np.append(vz, vzp)
        masses = np.append(masses, p_mass)

        # Add a False flag for each planet (indicating it is not a star)
        is_star = np.append(is_star, False)

    return x, y, z, vx, vy, vz, masses, is_star

#Generate the cluster
#Use the star generator and planet generator to get our complete cluster:
def cluster(N, R, mass_mean, mass_std, Q):
    # Generate stars
    x, y, z, vx, vy, vz, masses, is_star = generate_stars(N, R, mass_mean, mass_std, Q)

    # Select the star closest to 1 solar mass
    id = np.argmin(np.abs(masses - M_sun))

    # Generate planets around the chosen star
    x, y, z, vx, vy, vz, masses, is_star = planet(x, y, z, vx, vy, vz, masses, is_star, id)

    # Remove any net momentum introduced by the planets' random orbital phases.
    total_mass = np.sum(masses)
    vx -= np.sum(masses * vx) / total_mass
    vy -= np.sum(masses * vy) / total_mass
    vz -= np.sum(masses * vz) / total_mass

    return x, y, z, vx, vy, vz, masses, is_star
