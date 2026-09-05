#Derived physical quantities for compact binary systems

#Imports
import numpy as np

#Constants
SOLAR_MASS_KG = 1.98847e30
GRAVITATIONAL_CONSTANT = 6.67430e-11
SPEED_OF_LIGHT = 299_792_458.0

#Definitions
def chirp_mass(mass_ratio: float, total_mass: float) -> float:
    """Calculate chirp mass from q and total mass in solar masses."""
    if not 0 < mass_ratio <= 1 or total_mass <= 0:
        raise ValueError("mass_ratio must be in (0, 1] and total_mass must be positive")
    return float((mass_ratio / (1 + mass_ratio) ** 2) ** (3 / 5) * total_mass)


def component_masses(mass_ratio: float, total_mass: float) -> tuple[float, float]:
    """Return primary and secondary masses in solar masses."""
    if not 0 < mass_ratio <= 1 or total_mass <= 0:
        raise ValueError("mass_ratio must be in (0, 1] and total_mass must be positive")
    primary = total_mass / (1 + mass_ratio)
    return float(primary), float(mass_ratio * primary)


def schwarzschild_radii(mass_one: float, mass_two: float) -> tuple[float, float]:
    """Return component Schwarzschild radii in kilometres."""
    if min(mass_one, mass_two) <= 0:
        raise ValueError("component masses must be positive")
    factor = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS_KG / SPEED_OF_LIGHT**2 / 1000
    return factor * mass_one, factor * mass_two


def orbital_separation(mass_one: float, mass_two: float, gravitational_wave_period: float) -> float:
    """Estimate orbital separation in kilometres from a GW period."""
    if min(mass_one, mass_two, gravitational_wave_period) <= 0:
        raise ValueError("masses and period must be positive")
    angular_frequency = 2 * np.pi / (2 * gravitational_wave_period)
    separation = (GRAVITATIONAL_CONSTANT * SOLAR_MASS_KG * (mass_one + mass_two)
                  / angular_frequency**2) ** (1 / 3)
    return float(separation / 1000)