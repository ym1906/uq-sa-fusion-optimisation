# Read Me

## Demonstration of UQ tool suite

These tools have been developed to analyse the output of PROCESS Monte Carlo runs, but can analyse data from any software and source if it can be presented in a np.DataFrame format.

There is a suite of tools design to perform sensitivity analyses (SA), uncertainty quantification (UQ)

### Perform Regional Sensitivity Analysis
This looks for variables which cause convergence, caclulates a relative index for the most significance. 

In this example use-cases will be demonstrated.

![convergrence_sensitivity](<https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/Input%20Parameters%20Influencing%20Convergence_plot.png>)

### Calculate sensitivity for a given figure of merit
In this case, find the sensitivity towards the major radius, "rmajor". 

Uses rbd_fast method from Salib library. Higher number means more sensitivity.

Then filter for sensitivity above a given number.

![rmajor_sobol](https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/Sobol%20Indices%20for%20Major%20Radius_plot.png)

## Create a scatter plot of the results

- This creates a histogram color map of converged solutions (hist=True) and a scatter plot (scatter=True). 
- This can be used for visual identification of relationships, if there is a linear slant to the data it indicates a relationship exists
- Red on the color map indicates that more points fall in this region.
- You can plot an individual graph with "scatter" and a grid of scatter plots with "scatter_grid".

 ![scatter_plot](https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/tbrnmnrmajor-plot.png)
 ![scatter_grid](https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/scatter_gird.png)

## Create CDF plots
Plot the CDF of converged and unconverged samples, as well as the convergence rate for a given sampled parameter.

If there is a difference between the red and blue lines, this indicates that converged runs are coming from a different selection of input parameters to unconverged solutions (ie the figure of merit is sensitive to convergence).

As an example, compare the aspect ratio to the number of cycles in the CS coil.

![aspect_ecdf](https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/aspect-ecdf-plot.png)
![n_cycle_ecdf](https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/n_cycle_min-ecdf-plot.png)

## Regional Sensitivity Analysis 

We can investigate regional relationships between variables. For example, when the major radius is small, different things may be compromised to achieve a solution when then the major radius is large.

In this example, to achieve a high burn time the major radius must change from the typical size required for a lower burn time.

![rsatbrnmn](https://github.com/ym1906/uq-sa-fusion-design/blob/main/examples/plots/tbrnmn-rsa-plot.png)
