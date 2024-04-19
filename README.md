# Read Me

## Table of Contents

1. [UncertaintyData Class Documentation](#uncertaintydata-class-documentation)
   - [Key Components](#key-components)
     - [Initialization](#initialization)
     - [Data Handling](#data-handling)
     - [Convergence Analysis](#convergence-analysis)
     - [Sampled Variables and Figures of Merit](#sampled-variables-and-figures-of-merit)
   - [Methods](#methods)
   - [Additional Features](#additional-features)
2. [ConfidenceAnalysis Class Documentation](#introduction)
    - [Class Overview](#class-overview)
    - [Methods](#methods)
      - [Parameter Validation](#parameter-validation)
      - [Attribute Initialization](#attribute-initialization)
      - [Interval Probability Calculation](#interval-probability-calculation)
      - [Optimization and Data Preparation](#optimization-and-data-preparation)
      - [Additional Methods](#additional-methods)
    - [Usage](#usage)
    - [Examples](#examples)

## UncertaintyData Class Documentation

The `UncertaintyData` class is designed for collecting and analyzing output data from an `evaluate_uncertainties.py` tool. It merges HDF files containing uncertainty data, cleans, analyzes, and plots the data. Below, we’ll explore the key components and methods of this class.

### Key Components

#### Initialization

The constructor (`__init__`) takes the following parameters:

- `path_to_uq_data_folder`: Path to the folder containing uncertainty data.
- `figure_of_merit`: A metric used for analysis (e.g., performance, cost, etc.).
- `input_parameters`: A list of input parameter names.

#### Data Handling

- The class merges HDF files containing uncertainty data.
- Columns with the same value in every row are removed.
- The `sqsumsq` column is transformed using the natural logarithm.
- The input parameter names are stored in `self.input_names`.

#### Convergence Analysis

- The class identifies converged and unconverged runs based on an `ifail` column.
- If no failed runs are found, it assumes all runs are converged.

#### Sampled Variables and Figures of Merit

- Sampled variables for plotting are stored in `self.sampled_vars_to_plot`.
- The significance level (default 0.05) is used for plotting.
- The number of converged and unconverged runs is tracked.

### Methods

- `estimate_design_values(self, variables)`: Calculates mean values of sampled parameters as initial guesses. Assumes a uniform distribution for the parameters.
- `configure_data_for_plotting(self)`: Organizes UQ data into dataframes suitable for plotting. Plots either all sampled parameters or user-named parameters.
- `calculate_sensitivity(self, figure_of_merit)`: Computes sensitivity indices for converged UQ runs using the Salib `rbd_fast` method.
- `plot_rbd_si_indices(self)`: Calculates RBD FAST Sobol Indices and creates a plot. Displays sensitivity indices (S1) for different parameters related to fusion power.

### Additional Features

The class provides methods for regional sensitivity analysis, HDMR analysis, and ECDF plots.

## ConfidenceAnalysis Class Documentation

The `ConfidenceAnalysis` class is designed to perform interval analysis on uncertainty quantification (UQ) and Copula data. It calculates confidence intervals, optimizes configurations, and prepares data for plotting. This documentation provides an overview of the class, its methods, and usage guidelines.

## Introduction

The `ConfidenceAnalysis` class is part of a larger UQ and Copula analysis framework. It aims to provide insights into the confidence intervals of uncertain parameters and optimize configurations based on interval probabilities. By using this class, you can enhance your understanding of the uncertainty associated with your system.

## Class Overview

- **Class Name**: ConfidenceAnalysis
- **Purpose**: Interval analysis for UQ and Copula data
- **Attributes**:
  - `number_intervals`: Number of intervals for probability calculations
  - `weight_confidence`: Weighting factor favoring high confidence
  - `weight_overlap`: Weighting factor favoring no error overlap
  - `uq_data`: Uncertainty data for analysis
  - `input_names`: List of input variable names
  - `converged_data`: Converged data (real or synthetic)
  - `custom_data_point`: Custom data point for analysis
  - ... (other relevant attributes)

## Methods

### Parameter Validation

- `__init__(self, uq_data, input_names, ...)`: Initializes the class instance, validates input parameters, and sets attributes.

### Attribute Initialization

- `_sort_converged_data_by_variable(self, variable)`: Sorts the converged data by the specified variable.
- `_get_design_range_and_value(self, variable)`: Determines the design range and value for the given variable.

### Interval Probability Calculation

- `calculate_variable_probabilities(self, variable, number_intervals)`: Finds intervals in uncertain space with the highest rate of convergence. Calculates interval probabilities and confidence.

### Optimization and Data Preparation

- `modify_data(self, variable_data)`: Converts variable data into a format suitable for plotting graphs and tables. Computes joint probabilities and additional statistics.

### Additional Methods

- `_add_interval_column(self, dataframe, variable, design_range_intervals)`: Adds an interval column to the given DataFrame.
- `_calculate_metric(self, confidences, errors, weight_confidence, weight_overlap)`: Calculates a metric to evaluate the number of intervals.
- ... (other relevant methods)

## Usage

1. Instantiate the `ConfidenceAnalysis` class with appropriate parameters.
2. Call relevant methods to perform interval analysis and prepare data.
3. Visualize results using plots, tables, or other tools.

## Examples

Check the `examples` folder for a notebook implementation.
