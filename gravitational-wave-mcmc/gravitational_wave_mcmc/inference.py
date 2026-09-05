#Likelihoods and a lightweight Metropolis-Hastings sampler

#Imports
import numpy as np
from scipy.stats import norm
from .waveform import Waveform, scale_waveform

#Definitions
def log_likelihood(observed: np.ndarray, model: np.ndarray, noise_std: float) -> float:
    """Evaluate a Gaussian-noise log likelihood."""
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    residual = np.asarray(observed) - np.asarray(model)
    return float(-0.5 * np.sum((residual / noise_std) ** 2))


def log_prior(theta: np.ndarray, means: np.ndarray, standard_deviations: np.ndarray) -> float:
    """Evaluate independent normal priors for mass and distance."""
    theta = np.asarray(theta, dtype=float)
    means = np.asarray(means, dtype=float)
    standard_deviations = np.asarray(standard_deviations, dtype=float)
    if theta.shape != (2,) or means.shape != (2,) or standard_deviations.shape != (2,):
        raise ValueError("theta, means, and standard_deviations must contain mass and distance")
    if np.any(standard_deviations <= 0):
        raise ValueError("prior standard deviations must be positive")
    if np.any(theta <= 0):
        return -np.inf
    return float(np.sum(norm.logpdf(theta, loc=means, scale=standard_deviations)))


def log_posterior(theta: np.ndarray, observed: Waveform, reference: Waveform,
                  noise_std: float, reference_mass: float, reference_distance: float,
                  prior_means: np.ndarray, prior_standard_deviations: np.ndarray) -> float:
    """Evaluate the posterior for mass and distance parameters."""
    prior = log_prior(theta, prior_means, prior_standard_deviations)
    if not np.isfinite(prior):
        return -np.inf
    model = scale_waveform(observed.time, reference, reference_mass, reference_distance,
                           float(theta[0]), float(theta[1]))
    valid = np.isfinite(model)
    if not np.any(valid):
        return -np.inf
    return prior + log_likelihood(observed.strain[valid], model[valid], noise_std)


def metropolis_hastings(log_probability, initial: np.ndarray,
                        proposal_standard_deviations: np.ndarray, iterations: int,
                        rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw samples with a random-walk Metropolis-Hastings chain."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng() if rng is None else rng
    current = np.asarray(initial, dtype=float).copy()
    proposal_standard_deviations = np.asarray(proposal_standard_deviations, dtype=float)
    if current.ndim != 1 or proposal_standard_deviations.shape != current.shape:
        raise ValueError("initial and proposal standard deviations must have matching shapes")
    if np.any(proposal_standard_deviations <= 0):
        raise ValueError("proposal standard deviations must be positive")
    current_log_probability = float(log_probability(current))
    samples = np.empty((iterations, current.size))
    log_probabilities = np.empty(iterations)
    for index in range(iterations):
        proposed = current + rng.normal(0, proposal_standard_deviations)
        proposed_log_probability = float(log_probability(proposed))
        if np.log(rng.random()) < proposed_log_probability - current_log_probability:
            current = proposed
            current_log_probability = proposed_log_probability
        samples[index] = current
        log_probabilities[index] = current_log_probability
    return samples, log_probabilities