# Cancer Survival Analysis with Kaplan‑Meier and Cox Models

This project provides a complete framework for survival analysis on cancer patient data. It includes:

- **Synthetic data generation** for testing and demonstration.
- **Kaplan‑Meier survival curves** (overall and stratified by categorical variables).
- **Log‑rank tests** for comparing survival distributions between groups.
- **Cox proportional hazards model** to estimate hazard ratios for multiple covariates.
- **Visualization** of survival curves and hazard ratios.
- **Command‑line interface** for flexible usage.

The analysis is powered by the `lifelines` library, which implements robust survival analysis methods.

---

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Synthetic Data](#generate-synthetic-data)
  - [Analyze Existing Data](#analyze-existing-data)
  - [Full Pipeline](#full-pipeline)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Results Interpretation](#results-interpretation)
- [License](#license)

---

## Dataset

The script expects a CSV file with the following columns (names can be changed in `CONFIG`):

| Column Name      | Description                                      | Type        |
|------------------|--------------------------------------------------|-------------|
| `survival_time`  | Time to event (e.g., months)                     | numeric     |
| `event_observed` | 1 if event occurred (death), 0 if censored       | binary      |
| `age`            | Patient age at diagnosis                         | numeric     |
| `treatment`      | Type of treatment (e.g., Chemotherapy, Surgery)  | categorical |
| `cancer_stage`   | Cancer stage (e.g., Stage I, Stage II)           | categorical |

If no real data is provided, the script can generate synthetic data that mimics real‑world patterns.

---

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `lifelines` (for survival analysis)

All dependencies are listed in `requirements.txt`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cancer-survival-analysis.git
   cd cancer-survival-analysis
