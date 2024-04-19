# PROCESS_UQ

This repository contains some tools to perform UQ analysis on the output of Monte Carlo analysis with PROCESS.

PROCESS is a `Systems Code` which is used to design and optimise fusion power plants.

## Uncertainty Quantification Analysis

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
    Whprint("Failure Rate: ", uq_data.failure_probability,"+/-", uq_data.failure_cov)
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

##### Tips using copulas

You can fit "bounded or "unbounded" copulas. Each approach has some benefits and drawbacks.
The copula package tries to fit distributions to the data. You can modify the code to set the exact fit you want for a given distribution, and it's recommended to experiment with different approaches to see how it can effect the results. Try to reduce the number of parameters you sample (discard uninfluential parameters), and increase the number of samples you have to model.

###### Unbounded fit

Unbounted includes Gaussian Kernel Density Estimation (KDE). This means that the probability density function (PDF) is represented as a sum of Gaussian kernels, centred on each data point.

This technique is useful as it doesn't assume the shape of the distribution of the data, this means it can fit the data well.
However, there are some drawbacks:

- It's difficult to fit with few sample points.
- It's more resource heavy which means it's difficult to generate large numbers of synthetic samples (limits the number of parameters you can model and the number of samples you can generate).

To use an unbounded fit for a few sample points, you probably need at least 100 samples but the more the better. If you don't have enough samples the Copula package won't be able to fit a Gaussian KDE.

###### Bounded fit

Unbounded uses a parameterised fitting technique. This means that the Copula tool tries to select a predefined distribution to fit to the data (typically a Gaussian distribution, could be gamma or beta). It's less computationally expensive to do this, and the fit is parameterised so that it won't generate data outside the bounds of the sample set. This is beneficial if you are working with a large number of parameters and want to generate many sythentic samples. It's also easier to fit these distributions to few data points.
The downside is that the fit may not closely match the dataset and the results may not be truly representative (imagine fitting a gaussian to a gamma distribution).

###### Predictions using the copula

One of the reasons to use the copula is to predict from a few samples where the optimal space is. This can be used to quickly optimise a design space, without wasting time sampling. The idea is to use only a few converged runs (5-20) to quickly determine the optimal region of the design space.
