#Imports
import math

#Functions for probabilistic modelling
def saturating_probability(exposure: float, rate: float) -> float:
    return 1 - math.exp(-rate * exposure)


def positive_predictive_value(
    prevalence: float, sensitivity: float, false_positive_rate: float
) -> float:
    positive_rate = (sensitivity * prevalence) + (
        false_positive_rate * (1 - prevalence)
    )
    return (sensitivity * prevalence) / positive_rate


def main() -> None:
    exposure = 0.5
    rate = 1.6
    prevalence = 0.30
    sensitivity = 0.92
    false_positive_rate = 0.04
    standard_error = 0.025

    sample_size = math.ceil(0.5 * (1 - 0.5) / standard_error**2)
    ppv = positive_predictive_value(prevalence, sensitivity, false_positive_rate)

    print("Probabilistic Modelling Case Studies")
    print(f"Saturating probability at exposure=0.5: {saturating_probability(exposure, rate):.3f}")
    print(f"Positive predictive value: {ppv:.3f}")
    print(f"Conservative sample size for SE={standard_error}: {sample_size}")


if __name__ == "__main__":
    main()