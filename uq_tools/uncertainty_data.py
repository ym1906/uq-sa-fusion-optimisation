import pandas as pd
import os
from SALib.analyze import rbd_fast, rsa
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import LineString
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot, row
from bokeh.io import export_svgs, export_png, export_svg
from bokeh.models import ColumnDataSource, Range1d


class UncertaintyData:
    """The tool looks for hdf files containing uncertainty data, merges them,
    and has functions to clean, analyse, and plot the data.
    """

    def __init__(
        self,
        path_to_uq_data_folder,
        sampled_variables,
    ):
        self.path_in = path_to_uq_data_folder
        self.sampled_variables = sampled_variables
        self.image_export_path = path_to_uq_data_folder

    def load_data(self):
        self.uncertainties_df = merge_hdf_files(self.path_in)
        # Remove columns which have the same value in every row.
        self.unique_array = unique_cols(self.uncertainties_df)
        self.uncertainties_df = self.uncertainties_df.loc[:, ~self.unique_array]
        self.uncertainties_df["sqsumsq"] = np.log(self.uncertainties_df["sqsumsq"])

    def process_data(self):
        self.number_sampled_vars = len(self.sampled_variables)
        self.problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.sampled_variables,
        }
        # Drop the unnecessary levels from the columns
        self.uncertainties_df.columns = self.uncertainties_df.columns.droplevel(1)

    def separate_converged_unconverged(self):
        try:
            self.converged_df = self.uncertainties_df[
                self.uncertainties_df["ifail"] == 1.0
            ]
            self.unconverged_df = self.uncertainties_df[
                self.uncertainties_df["ifail"] != 1.0
            ]
        except KeyError as e:
            # Exception to handle the case where there are no failed runs.
            print(f"KeyError: {e}")
            self.converged_df = self.uncertainties_df
            self.unconverged_df = pd.DataFrame(
                data=None,
                columns=self.converged_df.columns,
                index=self.converged_df.index,
            )

    def set_image_export_path(self, path):
        """
        Sets the image export path.

        Args:
            path (str): The desired export path for images.
        """
        self.image_export_path = path

    def initialize_data(self):
        """Runs data processing functions in sequence."""
        self.load_data()
        self.process_data()
        self.separate_converged_unconverged()

    def initialize_plotting(self):
        self.converged_sampled_vars_df = self.converged_df[self.sampled_variables]
        self.unconverged_sampled_vars_df = self.unconverged_df[self.sampled_variables]
        # index values insignificant, 0.05 by default - subjective.
        self.number_of_converged_runs = len(self.converged_df.index)
        self.number_of_unconverged_runs = len(self.unconverged_df)

    def estimate_design_values(uq_dataframe, variables):
        """Find the mean values of sampled parameters as a guess of input initial value
        (assumes uniform distribution)"""
        design_values_df = uq_dataframe[variables]
        mean_values_df = design_values_df.mean(axis=0)

        return mean_values_df

    # def configure_data_for_plotting(self, variables_to_plot=None):
    #     """This function sorts the UQ data into dataframes for plotting with the hdf_to_scatter tool."""
    #     if variables_to_plot is None:
    #         variables_to_plot = self.converged_df.columns.tolist()
    #         variables_to_plot.remove("run_id")
    #         variables_to_plot.remove("ifail")
    #         variables_to_plot.remove("sqsumsq")
    #         print("Plotting all variables found in UQ dataframe.")
    #         print(variables_to_plot)
    #         self.plot_converged_df = self.converged_df[variables_to_plot]
    #         self.plot_unconverged_df = self.unconverged_df[variables_to_plot]
    #     else:
    #         print("Ploting user named parameters.")
    #         self.plot_converged_df = self.converged_df[variables_to_plot]
    #         self.plot_unconverged_df = self.unconverged_df[variables_to_plot]

    def filter_dataframe(self, dataframe, variables):
        """Filter a dataframe for given variables.

        :param variables: Figures of merit
        :type fom: str
        :return: Converged FOM DataFrame
        :rtype: pandas.Series
        """
        return dataframe[variables]

    def calculate_sensitivity(self, figure_of_merit, sampled_variables):
        """Calculate the sensitivity indices for a set of converged UQ runs.
        Uses the Salib rbd_fast analysis method.
        """
        converged_figure_of_merit_df = self.filter_dataframe(
            self.converged_df, figure_of_merit
        )
        sampled_variables_df = self.filter_dataframe(
            self.converged_df, sampled_variables
        )
        problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.sampled_variables,
        }
        sirbd_fast = rbd_fast.analyze(
            problem,
            sampled_variables_df.to_numpy(),
            converged_figure_of_merit_df.to_numpy(),
        )
        self.sensitivity_df = sirbd_fast.to_df()
        self.sensitivity_df = self.sensitivity_df.sort_values(by="S1", ascending=False)
        variable_names = self.sensitivity_df.index.values.tolist()
        indices = self.sensitivity_df["S1"]
        index_err = self.sensitivity_df["S1_conf"]

        return variable_names, indices, index_err

    def find_significant_parameters(self, sensitivity_data, significance_level):
        """Find the parameters above a given significance level.

        :param sensitivity_data: Dataframe with sensitivity indices
        :type sensitivity_data: pandas.DataFrame
        :param significance_level: Significance level (user defined)
        :type significance_level: float
        :return: Significant parameters
        :rtype: list
        """
        significant_df = sensitivity_data[sensitivity_data["S1"].ge(significance_level)]
        self.significant_conv_vars = significant_df.index.map(str).values
        return significant_df.index.map(str).values.tolist()

    def find_influential_conv_parameters(self):
        """Find the input parameters with a value above the significance level.
        self.convergence_rsa_result = pd.DataFrame()

        :return: _description_
        :rtype: _type_
        """
        significant_df = self.convergence_rsa_result[
            self.convergence_rsa_result["converged"].ge(self.significance_level)
        ]
        return significant_df.index.map(str).values

    def calculate_failure_probability(self):
        """Calculate the probability of failure and its coefficient of variation (C.O.V). This is defined as the number of failed runs
        divided by the number of converged runs.
        """
        num_converged = len(self.converged_df.index)
        num_sampled = len(self.uncertainties_df.index)
        self.failure_probability = round(1 - (num_converged) / num_sampled, 2)
        # The error on failure rate
        try:
            self.failure_cov = round(
                np.sqrt(
                    (1 - self.failure_probability)
                    / (self.failure_probability * num_sampled)
                ),
                2,
            )
            error_on_failure_probability = np.sqrt(
                (self.failure_probability * (1 - self.failure_probability))
                / num_sampled
            )
            # Set the confidence level (e.g., 95%)
            confidence_level = 0.95
            # Calculate the z-score for the desired confidence level
            z_score = 1.96  # for 95% confidence interval

            # Calculate the margin of error
            margin_of_error = z_score * error_on_failure_probability
            self.failure_cov = round(margin_of_error, 2)
        except ZeroDivisionError as e:
            # When the failure rate is 1
            self.failure_cov = 0.0

    def plot_sobol_indices(self, figure_of_merit, significance_level=None):
        """Calculate RBD FAST Sobol Indices and plot"""
        significance_level = significance_level or 0.05
        fig, ax = plt.subplots(1)
        # fig.set_size_inches(18, 5)

        ax.tick_params(labelsize=14)
        sensdf = self.sensitivity_df
        sensdf = sensdf.rename(process_variable_dict, axis=0)
        # x-axis
        sensdf.plot(
            kind="barh", y="S1", xerr="S1_conf", ax=ax, align="center", capsize=3
        )
        # y-axis
        ax.set_xlabel("Sensitivity Indices: " + figure_of_merit, fontsize=20)
        ax.set_ylabel("PROCESS parameter", fontsize=20)

        # in striped grey : not significant indices
        ax.fill_betweenx(
            y=[-0.5, len(self.sensitivity_df)],
            x1=0,
            x2=significance_level,
            color="grey",
            alpha=0.2,
            hatch="//",
            edgecolor="white",
        )
        plt.grid()
        # plt.savefig("plots/sensitivity_fom.svg", bbox_inches="tight")
        plt.show()

    def plot_sensitivity(
        self, indices, names, export_image=False, significance_level=None, title=None
    ):

        significance_level = significance_level or 0.05

        # Create a Bokeh figure
        p = figure(
            title=title,
            width=800,
            height=400,
            y_range=names,
        )
        # Create a ColumnDataSource for the data
        source = ColumnDataSource(
            {
                "x": indices,
                "names": names,
            }
        )
        # Add a shaded region of insignificance
        p.quad(
            top=len(indices),
            bottom=-0.5,
            left=0,
            right=significance_level,
            fill_color="grey",
            fill_alpha=0.2,
            line_color="white",
            legend_label="Region of Insignificance",
        )
        # Plot horizontal bars#
        p.hbar(
            y="names",
            left=0,
            right="x",
            height=0.8,
            source=source,
            fill_color="blue",
            line_color="white",
            legend_label="Significance Index",
            alpha=0.7,
        )

        # Customize the plot
        p.xaxis.axis_label = "Index"
        p.yaxis.axis_label = "PROCESS Parameter"
        p.legend.location = "top_right"
        p.legend.label_text_font_size = "12pt"

        show(p)
        if export_image:
            # Use the default directory of the script
            filename = self.image_export_path + "/crsa_plot.png"
            export_png(p, filename=filename)

    def convergence_study(self, n, sampled_variables, process_output):
        """This function is used to calculate RBD FAST sensitivities indices for a subset.
        It draws a random sample, of a given size, from the total dataset. This is used to
        create a study of the convergence of the indices for the input parameters.

        :param n: Number of samples to select
        :type n: int
        :param sampled_variables: Array of the sampled input parameters
        :type sampled_variables: numpy.array()
        :param process_output: Array of the figure of merit
        :type process_output: numpy.array()
        :return: Sensitivity Indices dict, length of the subset
        :rtype: dict, int
        """
        subset = np.random.choice(len(process_output), size=n, replace=False)
        problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.sampled_variables,
        }
        rbd_results = rbd_fast.analyze(
            problem, X=sampled_variables[subset], Y=process_output[subset]
        )
        return rbd_results, len(subset)

    def find_convergence_indices(self, step_size, figure_of_merit):
        """Calculate arrays of sensitivity indices for each input parameter to the figure of merit.
        Use the output array of indices to check for convergence (are they stable around a point)

        :param n: Calculate indices `n` times along the number of samples
        :type n: int
        :return: Array of arrays containing sensitivity indices for each sampled input
        :rtype: numpy.array()
        """
        converged_figure_of_merit_df = self.filter_dataframe(
            self.converged_df, figure_of_merit
        )
        sampled_vars_df = self.converged_sampled_vars_df

        indices_df_list = []
        for n in np.arange(
            start=50, stop=len(converged_figure_of_merit_df) + 1, step=step_size
        ):
            rbd_results, len_subset = self.convergence_study(
                n=n,
                sampled_variables=sampled_vars_df.to_numpy(),
                process_output=converged_figure_of_merit_df.to_numpy(),
            )
            rbd_results["samples"] = len_subset
            indices_df_list.append(rbd_results.to_df())
        indices_df = pd.concat(indices_df_list)

        return indices_df

    def plot_convergence_study(
        self, step_size, figure_of_merit, significance_level=None
    ):
        """Performs a convergence study and plots the results. Plots confidence levels only if
        final indices are greater than significance level.

        :param step_size: Number of samples to increment by when calculating sensitivity indices
        :type step_size: int
        """
        significance_level = significance_level or 0.05
        indices_df = self.find_convergence_indices(step_size, figure_of_merit)
        conv_fig, conv_ax = plt.subplots()
        conv_fig.set_size_inches(10, 6)
        conv_ax.set_ylim(ymin=-0.15, ymax=1)

        for name in self.sampled_variables:
            name_df = indices_df.loc[[name]]
            name_df.plot(
                x="samples",
                y="S1",
                label=process_variable_dict[name],
                ax=conv_ax,
                linewidth=3,
            )

            if name_df["S1"].iloc[-1] > significance_level:
                plt.fill_between(
                    name_df["samples"],
                    (name_df["S1"] - name_df["S1_conf"]),
                    (name_df["S1"] + name_df["S1_conf"]),
                    # color="tab:blue",
                    alpha=0.1,
                )

        conv_ax.fill_between(
            x=[indices_df["samples"].min(), indices_df["samples"].max()],
            y1=-0.15,
            y2=significance_level,
            color="grey",
            alpha=0.3,
            hatch="//",
            edgecolor="white",
            label="Not significant",
        )
        conv_ax.tick_params(labelsize=16)
        conv_ax.set_xlim(
            xmin=indices_df["samples"].min(), xmax=indices_df["samples"].max()
        )
        conv_ax.set_xlabel("Samples", fontdict={"fontsize": 20})
        conv_ax.set_ylabel("Sensitivity Index", fontdict={"fontsize": 20})
        conv_ax.legend(fontsize=12, borderpad=0.01, ncol=2)

        plt.grid()
        plt.savefig(
            "/home/graeme/data/uq_run_data/total_runs_2/uq_run_10p/plots/convergence.svg",
            bbox_inches="tight",
        )
        plt.show()

    def scatter(
        self,
        data,
        x_variable,
        y_variable,
        scatter=True,
        hist=True,
        bins=10,
    ):
        p = figure(title="Scatter Plot Comparison")

        # Extract data
        x = data[x_variable]
        y = data[y_variable]

        # Compute 2D histogram
        H, xe, ye = np.histogram2d(x=x, y=y, bins=bins)

        # Create an image plot
        if hist is True:
            p.image(
                image=[H.T],
                x=xe[0],
                y=ye[0],
                dw=xe[-1] - xe[0],
                dh=ye[-1] - ye[0],
                palette="Spectral11",
                alpha=0.6,
            )

        # Overlay scatter points
        if scatter is True:
            p.scatter(x=x, y=y, size=8, color="blue", alpha=0.5)

        # Customize the plot
        p.xaxis.axis_label = process_variable_dict[x_variable]
        p.yaxis.axis_label = process_variable_dict[y_variable]

        # Show the plot
        show(row(p))

    def scatter_grid(
        self,
        data,
        variables,
        bins=10,
        scatter=True,
        hist=True,
        height_width=250,
        export_image=False,
    ):
        # Create a grid of scatter plots
        plots = []
        for i, var1 in enumerate(variables):
            row_plots = []
            for j, var2 in enumerate(variables):
                if j >= i:
                    # Compute 2D histogram
                    H, xe, ye = np.histogram2d(x=data[var1], y=data[var2], bins=bins)

                    # Create an image plot
                    p = figure(
                        title=f"{var1} vs {var2}",
                        height=height_width,
                        width=height_width,
                    )
                    if hist is True:
                        p.image(
                            image=[H.T],
                            x=xe[0],
                            y=ye[0],
                            dw=xe[-1] - xe[0],
                            dh=ye[-1] - ye[0],
                            palette="Spectral11",
                            alpha=0.6,
                        )

                    # Overlay scatter points
                    if scatter is True:
                        p.scatter(
                            x=data[var1],
                            y=data[var2],
                            size=8,
                            color="blue",
                            alpha=0.5,
                        )
                    # Customize the plot
                    p.xaxis.axis_label = var1
                    p.yaxis.axis_label = var2
                    row_plots.append(p)
                else:
                    row_plots.append(None)

            plots.append(row_plots)

        # Create a grid layout
        grid = gridplot(plots)
        if export_image:
            filename = os.path.join(
                self.image_export_path,
                f"scatter-grid.png",
            )
            export_png(grid, filename=filename, height=height_width, width=height_width)
        # Show the plot
        show(grid)

    def convergence_regional_sensitivity_analysis(self, variable_names):
        """Regional sensitivity anlysis to find the parameters which influence a converged solution.
        Uses modified RSA technique from SALib."""
        variable_names = variable_names or self.sampled_variables
        figure_of_merit_df = self.uncertainties_df["sqsumsq"]
        sampled_vars_df = self.uncertainties_df[variable_names]
        convergence_problem = {
            "num_vars": len(sampled_vars_df),
            "names": variable_names,
        }
        convergence_rsa_result = rsa.analyze(
            problem=convergence_problem,
            X=sampled_vars_df.to_numpy(),
            Y=figure_of_merit_df.to_numpy(),
            bins=2,
            target="convergence",
            print_to_console=False,
            seed=1,
            mode="process",
        )
        convergence_rsa_result = convergence_rsa_result.to_df()
        convergence_rsa_result = convergence_rsa_result.T
        convergence_rsa_result.columns = ["converged", "unconverged"]
        convergence_rsa_result = convergence_rsa_result.sort_values(
            by="unconverged", ascending=False
        )

        self.convergence_rsa_result = convergence_rsa_result
        variable_names = convergence_rsa_result.index.values.tolist()
        indices = convergence_rsa_result["unconverged"]

        return variable_names, indices

    def regional_sensitivity_analysis(
        self,
        figure_of_merit,
        variables_to_sample,
        dataframe,
        bins=10,
        confidence_level=0.1,
    ):
        figure_of_merit_df = dataframe[figure_of_merit]
        sampled_vars_df = dataframe[variables_to_sample]
        problem = {
            "num_vars": len(sampled_vars_df),
            "names": variables_to_sample,
        }
        # Regional sensitivity analysis
        rsa_result = rsa.analyze(
            problem=problem,
            X=sampled_vars_df.to_numpy(),
            Y=figure_of_merit_df.to_numpy(),
            bins=bins,
            print_to_console=False,
            seed=1,
        )
        # Create a dataframe with the results with true values of FoM.
        rsa_res_df = pd.DataFrame(
            rsa_result,
            columns=rsa_result["names"],
            index=rsa_result["quants"][1 : 1 + bins],
        )

        # Plot the results
        fig, ax = plt.subplots()
        # Find the max RSA index, discard variables which are never above confidence level
        max_rsa = rsa_res_df.max()
        filtered_max_rsa = max_rsa[max_rsa > confidence_level]
        filtered_max_rsa = filtered_max_rsa.sort_values(ascending=False)
        rsa_res_df[filtered_max_rsa.index].plot(
            ylabel="Regional Influence Index", xlabel=figure_of_merit, ax=ax
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fancybox=True,
            shadow=True,
            ncol=5,
        )

        return filtered_max_rsa

    def ecdf_plot(self, figure_of_merit, num_steps=10, image_height_width=500):
        """Plot Empirical Cumulative distribution Functions for converged and unconverged samples.
        Additionally, plot the convergence rate and the design point value.

        :param figure_of_merit: Parameter to investigate
        :type figure_of_merit: str
        :param num_steps: Number of steps for the ECDF lines, default is 100
        :type num_steps: int
        """
        line_width = 10
        image_height_width
        # Arrange the data into df
        converged_fom_df = self.converged_df[figure_of_merit].to_numpy()
        uq_fom_df = self.uncertainties_df[figure_of_merit].to_numpy()
        # Calculate cumulative distribution functions
        ecdf_unconv = sm.distributions.ECDF(uq_fom_df)
        ecdf_conv = sm.distributions.ECDF(converged_fom_df)
        # Bin the data
        x = np.linspace(min(uq_fom_df), max(uq_fom_df), num_steps)
        ri_t = []
        y_unconv = ecdf_unconv(x)
        y_conv = ecdf_conv(x)

        # Plotting with Bokeh
        p = figure(
            x_axis_label=process_variable_dict[figure_of_merit],
            y_axis_label="Percentage of Samples",
            width=image_height_width,
            height=image_height_width,
            title="ECDF Plot",
        )
        # Plot the ecdf functions
        p.line(
            x,
            y_unconv,
            line_color="red",
            legend_label="Unconverged samples",
            line_width=line_width,
        )
        p.line(
            x,
            y_conv,
            line_color="blue",
            legend_label="Converged samples",
            line_width=line_width,
        )

        # Calculate rate of convergence for bins in x
        for d in range(len(x) - 1):
            n_c = len(
                self.converged_df[
                    self.converged_df[figure_of_merit].between(x[d], x[d + 1])
                ].index
            )
            n_t = len(
                self.uncertainties_df[
                    self.uncertainties_df[figure_of_merit].between(x[d], x[d + 1])
                ].index
            )
            if n_t == 0:
                n_t = 0.0000001
            ri = n_c / n_t
            ri_t.append(ri)

        # Plot convergence rate
        p.line(
            x[:-1],
            ri_t,
            line_color="orange",
            legend_label="Convergence rate",
            line_width=line_width,
        )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        show(p)

    # def ecdf_plot(self, figure_of_merit):
    #     """Plot Empirical Cumulative distribution Functions for converged and unconverged samples.
    #     Additionally, plot the convergence rate and the design point value.

    #     :param figure_of_merit: Parameter to investigate
    #     :type figure_of_merit: str
    #     """
    #     fig, ax1 = plt.subplots(1)
    #     plt.rcParams["axes.xmargin"] = 0
    #     plt.rcParams["axes.ymargin"] = 0
    #     ax2 = ax1.twinx()
    #     plt.style.library["tableau-colorblind10"]
    #     # Arrange the data into df
    #     converged_fom_df = self.converged_df[figure_of_merit].to_numpy()
    #     uq_fom_df = self.uncertainties_df[figure_of_merit].to_numpy()
    #     # Calculate cumulative distribution functions
    #     ecdf_unconv = sm.distributions.ECDF(uq_fom_df)
    #     ecdf_conv = sm.distributions.ECDF(converged_fom_df)
    #     # Bin the data
    #     x = np.linspace(min(uq_fom_df), max(uq_fom_df))
    #     ri_t = []
    #     y_unconv = ecdf_unconv(x)
    #     y_conv = ecdf_conv(x)
    #     # Plot the ecdf functions
    #     ax1.step(x, y_unconv, color="tab:red", label="Unconverged samples")
    #     ax1.step(x, y_conv, color="tab:blue", label="Converged samples")
    #     ax1.set_ylabel("Percentage of Samples", fontsize=20)
    #     ax1.set_xlabel(process_variable_dict[figure_of_merit], fontsize=20)
    #     # Calculate rate of convergence for bins in x
    #     for d in range(len(x) - 1):
    #         n_c = len(
    #             self.converged_df[
    #                 self.converged_df[figure_of_merit].between(x[d], x[d + 1])
    #             ].index
    #         )
    #         n_t = len(
    #             self.uncertainties_df[
    #                 self.uncertainties_df[figure_of_merit].between(x[d], x[d + 1])
    #             ].index
    #         )
    #         if n_t == 0:
    #             n_t = 0.0000001
    #         ri = n_c / n_t
    #         ri_t.append(ri)
    #     # Finds the edges of bins (must be a better way to do this section)
    #     h, edges = np.histogram(ri_t, bins=x)
    #     # Plot convergence rate
    #     ax1.stairs(ri_t, edges, color="tab:orange", label="Convergence rate")
    #     ax2.set_ylabel("Convergence Rate", fontsize=20)
    #     # Plot design point
    #     ypoint = 0.5
    #     if figure_of_merit == "kappa":
    #         ypoint = 1.0
    #     # copy curve line y coords and set to a constant
    #     lines = y_unconv.copy()
    #     lines[:] = ypoint

    #     # get intersection
    #     first_line = LineString(np.column_stack((x, y_unconv)))
    #     second_line = LineString(np.column_stack((x, lines)))
    #     intersection = first_line.intersection(second_line)
    #     ax1.legend()
    #     ax1.grid(axis="both")


def unique_cols(df):
    """Find columns which are not all the same value.

    :param df: DataFrame in question
    :type df: pandas.DataFrame
    :return: DataFrame containing with no constant columns
    :rtype: pandas.DataFrame
    """
    a = df.to_numpy()
    return (a[0] == a).all(0)


def merge_hdf_files(path_to_hdf):
    """Looks for uncertainty hdf files in the working folder and merges them into a
    single dataframe for analysis.

    :return: Uncertainties DataFrame, Parameters DataFrame
    :rtype: pandas.DataFrame, pandas.Dataframe
    """
    list_uncertainties_dfs = []
    for root, dirs, files in os.walk(path_to_hdf):
        for file in files:
            pos_hdf = root + os.sep + file
            if pos_hdf.endswith(".h5") and "uncertainties_data" in pos_hdf:
                extra_uncertainties_df = pd.read_hdf(pos_hdf)
                list_uncertainties_dfs.append(extra_uncertainties_df)
    return pd.concat(list_uncertainties_dfs)


process_variable_dict = {
    "walalw": "Max neutron wall-load (MW/m$^{2}$)",
    "kappa": "Plasma separatrix elongation",
    "triang": "Plasma separatrix triangularity",
    "ralpne": "Thermal alpha/electron density (%)",
    "etaech": "ECH wall plug to injector eff. (%)",
    "pinjalw": "Max injected power (MW)",
    "alstroh": "Allowable hoop stress in CS (Pa)",
    "coreradius": "Normalised core radius (m)",
    "sig_tf_wp_max": "Max sheer stress in TF coil (Pa)",
    "psepbqarmax": "Divertor protection (MWT/m)",
    "sig_tf_case_max": "Max stress in TF coil case (Pa)",
    "powfmw": "Fusion power (MW)",
    "etath": "Thermal to electric eff. (%)",
    "wallmw": "Neutron wall-load (MW/m$^{2}$)",
    "aspect": "Aspect ratio",
    "tbrnmn": "Minimum burn time (s)",
    "tape_thickness": "Tape thickness ()",
    "thicndut": "thicndut ()",
    "dhecoil": "dhecoil",
    "rmajor": "Major Radius (m)",
    "tmargtf": "Temperature margin TF coil",
    "dene": "dene",
    "ohcth": "ohcth",
    "beta": "beta",
    "betalim": "betalim",
    "n_cycle_min": "Minimum number of allowable stress cycles",
    "bt": "Magnetic field (T)",
    "te": "Electron density",
    "tfcth": "TF coil thickness (m)",
    "bigq": "Fusion gain",
    "bore": "Bore size (m)",
    "coheof": "Current end-of-flat-top",
    "cohbop": "Current beginning-of-flat-top",
    "kappa": "Eleongation",
    "fvsbrnni": "fvsbrnni",
    "itvar019": "itvar019",
    "itvar020": "itvar020",
    "jwptf": "jwptf",
    "vtfskv": "vtfskv",
    "vdalw": "vdalw",
    "tdmptf": "tdmptf",
    "thkcas": "thkcas",
    "thwcndut": "thwcndut",
    "fcutfsu": "fcutfsu",
    "cpttf": "cpttf",
    "plhthresh": "plhthresh",
    "tmargtf": "tmargtf",
    "tmargoh": "tmargoh",
    "oh_steel_frac": "Steel Fraction in CS coil",
    "pdivt": "pdivt",
    "powfmw": "Fusion Power (MW)",
}


def read_json(file):
    """Read and print a json file.


    :param file: Path to json

    :type file: str

    """
    df = pd.read_json(file, orient="split")
    print(df)
