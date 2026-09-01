# N-Body Simulation Cluster

A Python N-body simulator built as part of a 3rd year project for stellar clusters and planetary systems, using a 4th-order Hermite predictor-corrector integrator with adaptive (Aarseth) timestepping, used to investigate how the Solar System acted when in a clustered environment

## Key Features

- Stellar cluster generation with changeable initial conditions (`cluster_sim/generators.py`)
  - *Skills:* statistical sampling (log-normal mass distributions), constrained random generation, and enforcing physical constraints (virial equilibrium) on synthetic datasets
- Solar-system-style planet generation around a chosen host star
  - *Skills:* parameterizing real-world reference data (orbital elements) into reusable, vectorized `numpy` initial conditions
- 4th-order Hermite integrator with adaptive timestep control (`cluster_sim/integrator.py`)
  - *Skills:* numerical methods, algorithm implementation from first principles, and performance optimization via `numba` JIT compilation
- Orbital element tracking (eccentricity, angular momentum, period, semi-major axis, aphelion/perihelion)
  - *Skills:* deriving and validating physical quantities from raw time-series simulation output
- Energy diagnostics (kinetic, potential, total) for conservation checks
  - *Skills:* using invariants to validate model correctness, a core practice for verifying any simulation or numerical pipeline
- 3D trajectory visualization (`cluster_sim/visual.py`)
  - *Skills:* multi-dimensional data visualization, reference-frame transformations (e.g. centering on a body or the center of mass), and communicating results clearly through plots
- CSV export of simulation results (`cluster_sim/save.py`)
  - *Skills:* structuring and persisting large multi-body time-series datasets for downstream analysis

## Tech Stack

- **Numerical Integration:** `numpy`, `numba`
- **Visualization:** `matplotlib`
- **Interactive Demo:** `jupyter`

## Installation and Set Up

### Clone the Repository

```
git clone https://github.com/<your-username>/nbody-sim-cluster.git
cd nbody-sim-cluster
```

A fresh clone starts with empty `data/` folders; run `python main.py` or the demo notebook once to generate simulation output locally.

### Set Up a Virtual Environment (Optional)

```
python -m venv .venv
.venv\Scripts\activate  # On macOS/Linux use: source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

## Usage

Run the full cluster + planetary system simulation:
```
python main.py
```
This generates a cluster of stars with planets orbiting the star closest to one solar mass, integrates their motion, and saves CSV data and trajectory plots to `data/processed/test/`.

### Demo Notebook

`notebooks/solar_system_demo.ipynb` demonstrates the simulator on a simplified single Sun + 8 planets system, showing orbit trajectories and an energy conservation check. Select the project's virtual environment as the notebook kernel before running.

## Data Folders

The `data/processed/` and `data/raw/` folders are populated by running the simulation and are not tracked in Git (see `.gitignore`); only their folder structure is preserved via `.gitkeep` files.

## Project Structure

```
main.py                 Entry point: runs a cluster + planetary system simulation
cluster_sim/
  constants.py           Physical constants (G, AU, pc, M_sun, softening)
  generators.py          Initial condition generators for stars and planets
  dynamics.py            Acceleration, jerk, and energy calculations
  integrator.py          Hermite predictor-corrector integrator
  orbital.py             Orbital element calculations
  save.py                CSV/snapshot output
  visual.py              3D trajectory plotting
notebooks/
  solar_system_demo.ipynb  Demo notebook: Sun + 8 planets, orbit and energy plots
data/
  processed/             CSV outputs and plots from simulation runs (generated)
  raw/                   Optional raw snapshot files (generated)
```

## Project Origin and Academic Context

Unlike the two- or three-body problem, the N-body problem has no general analytical solution, so it must instead be simulated, with the code iteratively determining physical attributes (velocity, acceleration, etc.) at each stage. To accomplish this, we built a 4th-order Hermite scheme which computes derivatives up to the 4th order, ensuring accurate conservation of energy and momentum throughout the simulation. The code was built from scratch, using multiple academic papers as references.

Once built, the code was used to investigate why exoplanet systems observed elsewhere in the universe are so diverse, and why the Solar System appears comparatively abnormal, by trialling Solar System survival rates under different clustered star-forming environments. Several cluster conditions were tested, and survival rates were found for each of the planets.

## Academic Attribution & Acknowledgements

* **Institution:** Department of Physics & Astronomy, Cardiff University
* **Degree Program:** MPhys Integrated Masters in Astrophysics
* **Supervision:** Supervised by Professor Paul Clark