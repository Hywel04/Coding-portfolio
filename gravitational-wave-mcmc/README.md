# Gravitational-Wave MCMC

A modular Python project for analysing sampled compact-binary gravitational-wave signals and estimating source parameters with Bayesian inference. The workflow loads observed and simulated waveforms, aligns them at merger, estimates the noise floor, scales reference signals by mass and distance, and samples the posterior with a Metropolis-Hastings Markov chain.

The analysis is designed to run reproducibly offline. Waveform paths are explicit, and input files use a simple two-column CSV schema: `time (s)` and `strain`.

Completed as part of a Data Analysis module.

## Key Features

- **Waveform Loading:** Reads sampled gravitational-wave strain data from CSV files and validates the required columns.
- **Signal Preprocessing:** Aligns each waveform so the largest negative strain sample occurs at merger and estimates the mean and standard deviation of the noise region.
- **Interpolation and Scaling:** Interpolates reference waveforms onto observed time samples and scales their amplitude and time evolution for trial masses and distances.
- **Bayesian Parameter Estimation:** Evaluates Gaussian-noise likelihoods and normal priors before drawing posterior samples with a random-walk Metropolis-Hastings sampler.
- **Binary-System Physics:** Derives chirp mass, component masses, Schwarzschild radii, and orbital separation from posterior or trial parameters.
- **Notebook-Based Analysis:** Provides a staged workflow with diagnostic plots, residual analysis, convergence checks, and astrophysical interpretation.

## Tech Stack

- **Numerical Computing:** `numpy`
- **Data Loading:** `pandas`
- **Interpolation and Statistics:** `scipy`
- **Visualisation:** `matplotlib`
- **Development Environment:** Python 3.10 or newer, Jupyter or VS Code notebooks

## Installation & Setup

### Clone the Repository

```powershell
git clone https://github.com/Hywel04/Coding-portfolio.git
cd Coding-portfolio/gravitational-wave-mcmc
```

### Set Up a Virtual Environment (Optional)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

### Install Dependencies

Install the project dependencies from the requirements file:

```powershell
pip install -r requirements.txt
```

For editable package installation during development, use `python -m pip install -e .` instead.

## Usage

### Run the Basic Analysis Script

The main script loads `data/raw/Observedwaveform.csv`, aligns it to merger, estimates the noise statistics after `0.05` seconds, and prints a short summary:

```powershell
python main.py
```

### Use the Package Directly

```python
from gravitational_wave_mcmc import align_to_merger, estimate_noise, load_waveform

observed = align_to_merger(load_waveform("data/raw/Observedwaveform.csv"))
noise_mean, noise_std = estimate_noise(observed, start_time=0.05)

print(f"Noise mean: {noise_mean:.3e}")
print(f"Noise standard deviation: {noise_std:.3e}")
```

### Run the Notebooks

Install Jupyter if it is not already available, then launch the notebook interface from the repository root:

```powershell
python -m pip install jupyter
jupyter notebook
```

Open the notebooks in order to follow the complete analysis from preprocessing through posterior interpretation.

## Analysis Workflow

1. Load and inspect the observed waveform.
2. Align the signal to its merger sample and estimate the noise floor.
3. Interpolate a reference waveform onto the observed time grid.
4. Scale the reference signal for trial source masses and distances.
5. Evaluate the likelihood and prior, then sample the posterior with MCMC.
6. Use the inferred parameters to calculate derived binary-system properties.

## Project Structure

```text
gravitational-wave-mcmc/
├── data/
│   └── raw/                              # Observed, simulated, and reference CSV waveforms
├── notebooks/
│   ├── 01_signal_preprocessing.ipynb     # Merger alignment and noise estimation
│   ├── 02_interpolation_scaling.ipynb    # Interpolation checks and waveform scaling
│   └── 03_bayesian_mcmc_analysis.ipynb   # Posterior sampling and interpretation
├── gravitational_wave_mcmc/
│   ├── __init__.py                        # Public package interface
│   ├── waveform.py                        # Loading, alignment, interpolation, and scaling
│   ├── inference.py                       # Likelihoods, priors, and MCMC sampler
│   └── physics.py                         # Derived compact-binary quantities
├── main.py                                # Basic command-line analysis entry point
├── pyproject.toml                         # Package metadata and dependencies
├── requirements.txt                       # Runtime dependency list
└── README.md
```

## Data Format

Input waveform CSV files must contain the following columns:

```text
time (s), strain
```

The repository includes an observed waveform, mock waveforms, and a reference waveform in `data/raw/`. New compatible waveform files can be analysed by passing their path to `load_waveform`.

## Scientific Context

Gravitational-wave strain encodes information about the dynamics of compact binary systems such as merging black holes or neutron stars. This project uses a simplified waveform model and Bayesian parameter estimation to investigate how source mass and distance affect the observed signal. The posterior samples can then be used to summarise uncertainty and derive quantities such as chirp mass, component masses, Schwarzschild radii, and orbital separation.

The implementation is intended as an educational and research-analysis scaffold. It uses a lightweight Metropolis-Hastings sampler rather than a specialised production gravitational-wave inference framework, so sampler diagnostics and model assumptions should be considered when interpreting results.
