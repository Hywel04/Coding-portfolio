# Probabilistic Modelling

A reproducible Python project exploring how probability models can support decision-making under uncertainty. The project combines simple mathematical models with visual analysis and clear assumptions, making the results easy to inspect and extend.

## Key Features

- **Diminishing-Returns Modelling:** Compares a saturating response curve with a linear baseline.
- **Bayesian Updating:** Calculates the probability of a true positive after an imperfect screening result.
- **Sampling Error:** Estimates a conservative sample size for a target standard error in a binary proportion.
- **Sensitivity Thinking:** Shows how prevalence and test characteristics affect the interpretation of a positive result.
- **Reproducible Analysis:** Separates a lightweight command-line summary from the fuller notebook exploration.

## Tech Stack

- **Numerical Computing:** `numpy`
- **Mathematical Functions:** Python standard library `math`
- **Visualisation:** `matplotlib`
- **Interactive Analysis:** `jupyter`

## Installation & Setup

### Set Up a Virtual Environment (Optional)

```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1  # Windows PowerShell
# On macOS/Linux: source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Command-Line Summary

```bash
python main.py
```

This prints representative outputs for the saturating model, Bayesian screening calculation, and conservative sample-size estimate.

### Run the Notebook

```bash
jupyter notebook notebooks/probabilistic_modelling_case_studies.ipynb
```

The notebook provides the complete workflow, including plots, intermediate calculations, assertions, and discussion of model limitations.

## Project Structure

```text
probabilistic-modelling/
├── notebooks/
│   └── probabilistic_modelling_case_studies.ipynb  # Full analysis and plots
├── main.py                                         # CLI summary
├── README.md
└── requirements.txt
```

## Modelling Context

The examples are purely hypothetical. They demonstrate transferable data-science techniques rather than making claims about a real population, organisation, product, or diagnostic system. The project was repurposed from an piece of continual assessment for a data science module into a standalone portfolio project.
