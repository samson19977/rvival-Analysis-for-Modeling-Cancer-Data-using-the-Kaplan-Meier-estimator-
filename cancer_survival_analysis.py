#!/usr/bin/env python3
"""
Cancer Survival Analysis using Kaplan-Meier and Cox Proportional Hazards

This script performs survival analysis on cancer patient data.
It can generate synthetic data or load real data from a CSV file.
The analysis includes Kaplan-Meier survival curves, log-rank tests,
and Cox proportional hazards modeling.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# -------------------- Configuration --------------------
CONFIG = {
    # Data
    'data_path': Path('data/cancer_survival.csv'),   # path to real data; if None, generate synthetic
    'synthetic_n': 200,                               # number of patients for synthetic data
    'synthetic_seed': 42,
    'survival_scale': 12,                             # average survival in months (exponential)
    'event_prob': 0.7,                                # proportion of observed events

    # Analysis
    'time_col': 'survival_time',
    'event_col': 'event_observed',
    'categorical_cols': ['treatment', 'cancer_stage'],
    'continuous_cols': ['age'],

    # Output
    'plots_dir': Path('./plots'),
    'results_dir': Path('./results'),
}

# Create directories
CONFIG['plots_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# -------------------- Data Generation / Loading --------------------
def generate_synthetic_data(n, seed, survival_scale, event_prob):
    """Generate synthetic cancer survival data."""
    np.random.seed(seed)
    survival_times = np.random.exponential(scale=survival_scale, size=n)
    event_observed = np.random.binomial(1, event_prob, size=n)
    age = np.random.randint(30, 80, size=n)
    treatment = np.random.choice(['Chemotherapy', 'Radiation', 'Surgery'], size=n)
    cancer_stage = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], size=n)

    data = pd.DataFrame({
        'survival_time': survival_times,
        'event_observed': event_observed,
        'age': age,
        'treatment': treatment,
        'cancer_stage': cancer_stage
    })
    return data

def load_real_data(file_path):
    """Load survival data from CSV."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Check required columns
    required = [CONFIG['time_col'], CONFIG['event_col']] + CONFIG['continuous_cols'] + CONFIG['categorical_cols']
    missing = [col for col in required if col not in data.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    # Ensure categorical columns are string type
    for col in CONFIG['categorical_cols']:
        data[col] = data[col].astype(str)

    return data


# -------------------- Survival Analysis Functions --------------------
def plot_kaplan_meier(data, time_col, event_col, group_col=None, save_path=None):
    """Plot Kaplan-Meier survival curves, optionally stratified by a group."""
    plt.figure(figsize=(10, 6))

    if group_col is None:
        # Overall curve
        kmf = KaplanMeierFitter()
        kmf.fit(data[time_col], event_observed=data[event_col], label='Overall')
        kmf.plot_survival_function()
        plt.title('Kaplan-Meier Survival Curve (Overall)')
        plt.ylabel('Survival Probability')
    else:
        # Stratified by group
        for group in data[group_col].unique():
            subset = data[data[group_col] == group]
            kmf = KaplanMeierFitter()
            kmf.fit(subset[time_col], event_observed=subset[event_col], label=group)
            kmf.plot_survival_function()
        plt.title(f'Kaplan-Meier Survival Curves by {group_col}')
        plt.ylabel('Survival Probability')
        plt.legend()

    plt.xlabel('Time (Months)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Kaplan-Meier plot saved to {save_path}")
    plt.show()

def logrank_pairwise(data, time_col, event_col, group_col):
    """Perform pairwise log-rank tests between groups."""
    groups = data[group_col].unique()
    results = {}
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            subset1 = data[data[group_col] == g1]
            subset2 = data[data[group_col] == g2]
            result = logrank_test(
                subset1[time_col], subset2[time_col],
                event_observed_A=subset1[event_col],
                event_observed_B=subset2[event_col]
            )
            results[f"{g1} vs {g2}"] = result.p_value
    return results

def fit_cox_model(data, time_col, event_col, categorical_cols, continuous_cols):
    """Fit Cox proportional hazards model."""
    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    # Ensure all columns are numeric
    data_encoded = data_encoded.astype(float)

    cph = CoxPHFitter()
    cph.fit(data_encoded, duration_col=time_col, event_col=event_col)
    return cph

def plot_cox_results(cph, save_path=None):
    """Plot hazard ratios from Cox model."""
    plt.figure(figsize=(8, 6))
    cph.plot(hazard_ratios=True)
    plt.title('Hazard Ratios from Cox Proportional Hazards Model')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Cox hazard ratios plot saved to {save_path}")
    plt.show()


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Cancer Survival Analysis')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generate', 'analyze', 'full'],
                        help='Mode: generate synthetic data, analyze existing data, or full (generate+analyze)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data CSV (if analyze mode)')
    parser.add_argument('--output_prefix', type=str, default='survival_analysis',
                        help='Prefix for output files')
    args = parser.parse_args()

    if args.mode == 'generate':
        data = generate_synthetic_data(
            CONFIG['synthetic_n'], CONFIG['synthetic_seed'],
            CONFIG['survival_scale'], CONFIG['event_prob']
        )
        output_path = CONFIG['data_path']
        data.to_csv(output_path, index=False)
        logger.info(f"Synthetic data generated and saved to {output_path}")
        # Optionally show summary
        print(data.head())
        print("\nData Summary:")
        print(data.describe(include='all'))

    elif args.mode == 'analyze':
        if args.data_path is None:
            logger.error("Please provide --data_path for analysis mode.")
            sys.exit(1)
        data = load_real_data(Path(args.data_path))
        logger.info(f"Loaded data from {args.data_path} with shape {data.shape}")

        # Perform Kaplan-Meier overall
        plot_kaplan_meier(data, CONFIG['time_col'], CONFIG['event_col'],
                          save_path=CONFIG['plots_dir'] / f"{args.output_prefix}_km_overall.png")

        # Kaplan-Meier by cancer stage
        plot_kaplan_meier(data, CONFIG['time_col'], CONFIG['event_col'], group_col='cancer_stage',
                          save_path=CONFIG['plots_dir'] / f"{args.output_prefix}_km_by_stage.png")

        # Log-rank test between all stages
        pairwise_p = logrank_pairwise(data, CONFIG['time_col'], CONFIG['event_col'], group_col='cancer_stage')
        print("\nLog-rank pairwise p-values (Stage groups):")
        for comp, p in pairwise_p.items():
            print(f"{comp}: p = {p:.4f}")

        # Fit Cox model
        cph = fit_cox_model(data, CONFIG['time_col'], CONFIG['event_col'],
                            CONFIG['categorical_cols'], CONFIG['continuous_cols'])
        print("\nCox Proportional Hazards Model Summary:")
        cph.print_summary()

        # Save model summary to file
        with open(CONFIG['results_dir'] / f"{args.output_prefix}_cox_summary.txt", 'w') as f:
            f.write(cph.summary.to_string())
        logger.info(f"Cox model summary saved to {CONFIG['results_dir'] / f'{args.output_prefix}_cox_summary.txt'}")

        # Plot hazard ratios
        plot_cox_results(cph, save_path=CONFIG['plots_dir'] / f"{args.output_prefix}_cox_hazard_ratios.png")

    elif args.mode == 'full':
        # Generate synthetic data first
        data = generate_synthetic_data(
            CONFIG['synthetic_n'], CONFIG['synthetic_seed'],
            CONFIG['survival_scale'], CONFIG['event_prob']
        )
        logger.info("Synthetic data generated.")
        print(data.head())
        # Save the data (optional)
        data.to_csv(CONFIG['data_path'], index=False)
        logger.info(f"Data saved to {CONFIG['data_path']}")

        # Now analyze it
        # Kaplan-Meier overall
        plot_kaplan_meier(data, CONFIG['time_col'], CONFIG['event_col'],
                          save_path=CONFIG['plots_dir'] / f"{args.output_prefix}_km_overall.png")
        # By stage
        plot_kaplan_meier(data, CONFIG['time_col'], CONFIG['event_col'], group_col='cancer_stage',
                          save_path=CONFIG['plots_dir'] / f"{args.output_prefix}_km_by_stage.png")
        # Log-rank pairwise
        pairwise_p = logrank_pairwise(data, CONFIG['time_col'], CONFIG['event_col'], group_col='cancer_stage')
        print("\nLog-rank pairwise p-values (Stage groups):")
        for comp, p in pairwise_p.items():
            print(f"{comp}: p = {p:.4f}")
        # Cox model
        cph = fit_cox_model(data, CONFIG['time_col'], CONFIG['event_col'],
                            CONFIG['categorical_cols'], CONFIG['continuous_cols'])
        print("\nCox Proportional Hazards Model Summary:")
        cph.print_summary()
        with open(CONFIG['results_dir'] / f"{args.output_prefix}_cox_summary.txt", 'w') as f:
            f.write(cph.summary.to_string())
        plot_cox_results(cph, save_path=CONFIG['plots_dir'] / f"{args.output_prefix}_cox_hazard_ratios.png")

if __name__ == '__main__':
    main()
