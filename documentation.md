# Uncertainty Quantification Analysis

This guide explains how to use the Uncertaintiy Quantification (UQ) analysis tool.

## Motivation

When I started working on this tool I wanted to answer some questions:

1. How can we evaluate the "reliability" of a power plant design?
    1b. Is the design in a "densly populated" region of parameter space?
    1c. If not, should it be?
    1d. What do you need to optimise to get it there?
2. How can we evaluate the "performance recovery" of a power plant design?
    2b. What does PROCESS change to ensure the minimum performance requirements are met?

These questions can be summarised as:

1. Is our design reliabile? (Which uncertain values drive PROCESS convergence?)
2. Why is our design the way it is?

## Methodology

PROCESS works with an optimiser and a solver.
The solver tries to find a solution to the constraint equations posed in the input file.
The optimiser tries to find the optimum solution subject to a figure of merit, typically `major radius`.
We want to understand the internal logic of the PROCESS optimiser, what parameters does it trade to find an optimised and converged solution.
We can use that information to evaluate the concept and see what is being "traded" to achieve a good design.

### Procedure

#### Perform a Monte Carlo run: use PROCESS `evaluate_uncertainties.py` to generate some Monte Carlo data

    * This should give you output `uncertainties_data.h5` (and optionally `scoping_data.h5`)
    * `uncertainties_data.h5` should be data for the complete MC run. It contains all the parameters in the MFILE dumped as h5.
    * `scoping_data.h5` is a "scoping" data set. By default this is written after 20 converged runs. Allows you to peak at the data without waiting for a full run to complete.

#### Create a notebook and import tools

These tools are designed to be used with a jupytr notebook. Create a notebook and import the classes you'd like to use. Create instances of those classes.

    ```python
    proj_dir = "/path/to/your/output/"
    figure_of_merit = "rmajor"
    uq_data = UncertaintyData(proj_dir, figure_of_merit,use_scoping_data=False)
    ```

#### Compile the data and perform analysis

`UncertaintyData` is a class which looks for files in the format `uncertainties_data.h5` and `scoping_data.h5` (and variants on the name) and compiles them into a dataframe. It then has various functions to clean, sort and analyse the data.

    ```python
    uq_data.calculate_sensitivity(figure_of_merit)
    uq_data.calculate_reliability()
    print("Number of samples: ", len(uq_data.uncertainties_df))
    print("Number of converged runs: ", uq_data.number_of_converged_runs)
    print("Reliability: ", uq_data.reliability_index)
    ```

    Perform a regional sensitivity analysis to determine which parameters are driving PROCESS convergence:

    ```python
    uq_data.convergence_regional_sensitivity_analysis(uq_data.input_names)
    uq_data.plot_sumsq_sensitivity()
    ```

    Retreive the values which are significant:

    ```python
    uq_data.convergence_regional_sensitivity_analysis(uq_data.input_names)
    uq_data.plot_sumsq_sensitivity()
    significant_conv_vars = uq_data.find_significant_parameters(uq_data.sumsq_sensitivity_df,"unconverged",0.05).tolist()
    ```

#### Further Analysis and Visualisation (Copulas)

A copula is a multivariate cumulative distribution function whcih is used to model the correlation between random variables.
A copula class has been created to help analyse the Monte Carlo output.
