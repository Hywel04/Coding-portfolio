#Loading, cleaning, and transforming sampled waveform data

#Imports
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#Definitions
@dataclass(frozen=True)
class Waveform:
    """A waveform represented by time in seconds and strain samples."""

    time: np.ndarray
    strain: np.ndarray

    def __post_init__(self) -> None:
        if self.time.ndim != 1 or self.strain.ndim != 1:
            raise ValueError("time and strain must be one-dimensional arrays")
        if self.time.size != self.strain.size or self.time.size == 0:
            raise ValueError("time and strain must have the same non-zero length")


def load_waveform(path: str | Path) -> Waveform:
    """Load a CSV waveform with ``time (s)`` and ``strain`` columns."""
    frame = pd.read_csv(path)
    required = {"time (s)", "strain"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing waveform columns: {sorted(missing)}")
    return Waveform(frame["time (s)"].to_numpy(float), frame["strain"].to_numpy(float))


def align_to_merger(waveform: Waveform) -> Waveform:
    """Shift time so the largest negative strain sample occurs at zero."""
    merger_index = int(np.argmin(waveform.strain))
    return Waveform(waveform.time - waveform.time[merger_index], waveform.strain.copy())


def estimate_noise(waveform: Waveform, start_time: float) -> tuple[float, float]:
    """Return mean and standard deviation after ``start_time``."""
    noise = waveform.strain[waveform.time >= start_time]
    if noise.size == 0:
        raise ValueError("start_time does not select any waveform samples")
    return float(np.mean(noise)), float(np.std(noise))


def interpolate_waveform(reference: Waveform, time: np.ndarray) -> np.ndarray:
    """Interpolate a reference waveform onto ``time`` samples."""
    interpolator = interp1d(reference.time, reference.strain, bounds_error=False, fill_value=np.nan)
    return np.asarray(interpolator(time), dtype=float)


def scale_waveform(time: np.ndarray, reference: Waveform, reference_mass: float,
                   reference_distance: float, mass: float, distance: float) -> np.ndarray:
    """Scale a reference waveform to a trial mass and distance."""
    if min(reference_mass, reference_distance, mass, distance) <= 0:
        raise ValueError("masses and distances must be positive")
    reference_time = np.asarray(time) * reference_mass / mass
    reference_strain = interpolate_waveform(reference, reference_time)
    return mass / reference_mass * reference_distance / distance * reference_strain