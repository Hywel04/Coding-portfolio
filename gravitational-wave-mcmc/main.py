# Main script for analyzing observed gravitational waveforms

#Imports
from pathlib import Path
from gravitational_wave_mcmc import align_to_merger, estimate_noise, load_waveform


DATA_PATH = Path(__file__).parent / "data" / "raw" / "Observedwaveform.csv"

#Definitions
def main() -> None:
    observed = align_to_merger(load_waveform(DATA_PATH))
    noise_mean, noise_std = estimate_noise(observed, start_time=0.05)
    print(f"Samples: {observed.time.size}")
    print(f"Aligned time range: {observed.time.min():.4f} to {observed.time.max():.4f} s")
    print(f"Noise mean: {noise_mean:.3e}")
    print(f"Noise standard deviation: {noise_std:.3e}")


if __name__ == "__main__":
    main()