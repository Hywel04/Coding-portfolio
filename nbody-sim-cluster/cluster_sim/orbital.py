#Contains all the orbital-related functions for the stellar cluster simulation

#Imports
import numpy as np
from .constants import G, epsilon
from numba import jit

# Calculate orbital properties following Kroupa et al.
@jit(nopython=True)
def kroupa(x, y, z, vx, vy, vz, masses, is_star):
    N = len(masses)
    ecc = []   # Eccentricity list
    ang = []   # Angular momentum list
    p = []     # Period list
    al = []    # Semi-major axis list
    eb_list = []  # Binding energy list (for each body)
    
    for i in range(N):
        if not is_star[i]:  # If i is a planet (or secondary body)
            best_eb = 0.0  # Only accept bound pairs
            best_j = -1  # Index of most bound partner

            for j in range(N):
                if is_star[j] and j != i:
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dz = z[j] - z[i]
                    dxv = vx[j] - vx[i]
                    dyv = vy[j] - vy[i]
                    dzv = vz[j] - vz[i]

                    softened_radius = np.sqrt(dx**2 + dy**2 + dz**2 + epsilon**2)
                    speed = dxv**2 + dyv**2 + dzv**2
                    msys = masses[i] + masses[j]
                    reducedm = (masses[i] * masses[j]) / msys

                    # Use the softened potential for pair selection.
                    eb = (0.5 * reducedm * speed) - ((G * masses[i] * masses[j]) / softened_radius)

                    # Track the most negative binding energy among bound pairs
                    if eb < best_eb:
                        best_eb = eb
                        best_j = j

            if best_j == -1:
                continue  # No bound star found

            j = best_j  # Use most bound star index

            # Recalculate orbital properties with most bound star
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            dxv = vx[j] - vx[i]
            dyv = vy[j] - vy[i]
            dzv = vz[j] - vz[i]

            radius = np.sqrt(dx**2 + dy**2 + dz**2)
            if radius == 0.0:
                continue
            speed = dxv**2 + dyv**2 + dzv**2
            dot = dx * dxv + dy * dyv + dz * dzv

            msys = masses[i] + masses[j]
            reducedm = (masses[i] * masses[j]) / msys
            # Classical orbital elements use the unsoftened Newtonian potential.
            orbital_eb = (0.5 * reducedm * speed) - ((G * masses[i] * masses[j]) / radius)
            if orbital_eb >= 0.0:
                continue
            a = -1 * ((G * masses[i] * masses[j]) / (2 * orbital_eb))
            e = np.sqrt((1 - (radius / a))**2 + (dot**2 / (a * G * msys)))
            an = masses[i] * masses[j] * np.sqrt(G * (1 / msys) * a * (1 - e**2))
            period = np.sqrt((4 * np.pi**2 * a**3) / (G * msys)) / 86400

            # Store results for the current planet (or secondary body)
            ecc.append(e)
            ang.append(an)
            p.append(period)
            al.append(a)
            eb_list.append(orbital_eb)  # Store the binding energy for this body

    return al, ecc, ang, p, eb_list