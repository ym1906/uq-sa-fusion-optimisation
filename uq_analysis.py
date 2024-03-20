import pandas as pd
import os
import re
from dataclasses import dataclass, asdict
from SALib.analyze import rbd_fast, hdmr, rsa
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D
from pylab import figure
from shapely.geometry import LineString
from copulas.visualization import (
    scatter_2d,
    compare_2d,
    compare_3d,
)
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate import (
    ParametricType,
    BoundedType,
    Univariate,
    GaussianUnivariate,
    GaussianKDE,
    BetaUnivariate,
    GammaUnivariate,
)
from bokeh.palettes import (
    PuBu5,
    PuRd5,
    PuBuGn5,
)
import python_fortran_dicts
from bokeh.plotting import figure, show, from_networkx
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    DataTable,
    TableColumn,
    BoxAnnotation,
    HBar,
    HTMLTemplateFormatter,
    Range1d,
    MultiLine,
    Circle,
    Whisker,
    TapTool,
    NodesAndLinkedEdges,
    BoxSelectTool,
    LabelSet,
    ResetTool,
)
from bokeh.io import export_svgs
import numpy as np
from itertools import combinations
import warnings


class UncertaintyData:
    """Collects and analyses the output from the evaluate_uncertainties.py tool. Supply an instance with the input
    folder for the run data. The tool looks for hdf files containing uncertainty data, merges them, and has functions
    to clean, analyse, and plot the data."""

    def __init__(
        self,
        path_to_uq_data_folder,
        figure_of_merit,
        input_parameters,
        use_scoping_data=False,
    ):
        self.path_in = path_to_uq_data_folder
        self.figure_of_merit = figure_of_merit

        # self.uncertainties_df, self.sampled_vars_df = self.merge_hdf_files(
        #     use_scoping_data
        # )
        self.uncertainties_df = self.merge_hdf_files()

        # Remove columns which have the same value in every row.
        self.unique_array = unique_cols(self.uncertainties_df)
        self.uncertainties_df = self.uncertainties_df.loc[:, ~self.unique_array]
        self.uncertainties_df["sqsumsq"] = np.log(self.uncertainties_df["sqsumsq"])
        self.input_names = input_parameters
        self.itv = [
            "bt",
            "te",
            "beta",
            "dene",
            "tfcth",
            "wallmw",
            "ohcth",
            "bigq",
            "bore",
            "betalim",
            "coheof",
            "cohbop",
            "kappa",
            # "gapoh",
            "fvsbrnni",
            "itvar019",
            "itvar020",
            "jwptf",
            "vtfskv",
            "vdalw",
            "tdmptf",
            "thkcas",
            "thwcndut",
            "fcutfsu",
            "cpttf",
            # "gapds",
            "plhthresh",
            "tmargtf",
            "tmargoh",
            "oh_steel_frac",
            "pdivt",
            "powfmw",
            "rmajor",
        ]
        # self.input_names.extend(itv)
        self.number_sampled_vars = len(self.input_names)
        self.problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.input_names,
        }

        # Drop the unnecessary levels from the columns
        self.uncertainties_df.columns = self.uncertainties_df.columns.droplevel(1)
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
        self.sampled_vars_to_plot = []
        self.plot_names = []
        self.number_sampled_vars = len(self.input_names)
        self.converged_sampled_vars_df = self.converged_df[self.input_names]
        self.unconverged_sampled_vars_df = self.unconverged_df[self.input_names]
        self.unconverged_fom_df = self.uncertainties_df[self.figure_of_merit]
        # Significance level is used for plotting, indicates level below which we consider
        # index values insignificant, 0.05 by default - subjective.
        self.significance_level = 0.05
        self.number_of_converged_runs = len(self.converged_df.index)
        self.number_of_unconverged_runs = len(self.unconverged_df)

        # Using a dict for converting param names for plotting
        self.name_dict = {
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
            "rmajor": "major radius (m)",
            "tmargtf": "Temperature margin TF coil",
            "dene": "dene",
            "ohcth": "ohcth",
            "beta": "beta",
            "betalim": "betalim",
        }

    def estimate_design_values(self, variables):
        """Find the mean values of sampled parameters as a guess of input initial value
        (assumes uniform distribution)"""
        design_values_df = self.uncertainties_df[variables]
        mean_values_df = design_values_df.mean(axis=0)

        return mean_values_df

    def configure_data_for_plotting(self):
        """This function sorts the UQ data into dataframes for plotting with the hdf_to_scatter tool."""
        if len(self.sampled_vars_to_plot) == 0:
            print("Plotting all sampled parameters")
            self.plot_names.extend(self.input_names)
            self.plot_names.extend([self.figure_of_merit])

            self.plot_converged_df = self.converged_df[self.plot_names]
            self.plot_unconverged_df = self.unconverged_df[self.plot_names]
        else:
            print("Ploting user named parameters")
            self.plot_names.extend(self.sampled_vars_to_plot)
            self.plot_names.extend([self.figure_of_merit])

            self.plot_converged_df = self.converged_df[self.plot_names]
            self.plot_unconverged_df = self.unconverged_df[self.plot_names]

    def get_fom_converged_df(self, figure_of_merit):
        """Get the Figure of Merit (fom) from the dataframe containing converged runs.

        :param fom: Figure of Merit
        :type fom: str
        :return: Converged FOM DataFrame
        :rtype: pandas.Series
        """
        return self.converged_df[figure_of_merit]

    def calculate_sensitivity(self, figure_of_merit):
        """Calculate the sensitivity indices for a set of converged UQ runs.
        Uses the Salib rbd_fast analysis method.
        """
        converged_figure_of_merit_df = self.get_fom_converged_df(figure_of_merit)
        sampled_vars_df = self.converged_sampled_vars_df
        self.problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.input_names,
        }
        sirbd_fast = rbd_fast.analyze(
            self.problem,
            sampled_vars_df.to_numpy(),
            converged_figure_of_merit_df.to_numpy(),
        )
        self.sensitivity_df = sirbd_fast.to_df()
        self.sensitivity_df = self.sensitivity_df.sort_values(by="S1", ascending=False)
        self.pdf = []

    def find_significant_parameters(
        self, sensitivity_data, data_column, significance_level
    ):
        """Find the parameters above a given significance level.

        :param sensitivity_data: Dataframe with sensitivity indices
        :type sensitivity_data: pandas.DataFrame
        :param significance_level: Significance level (user defined)
        :type significance_level: float
        :return: Significant parameters
        :rtype: list
        """
        significant_df = sensitivity_data[
            sensitivity_data[data_column].ge(significance_level)
        ]
        self.significant_conv_vars = significant_df.index.map(str).values
        return significant_df.index.map(str).values

    def find_influential_conv_parameters(self):
        """Find the input parameters with a value above the significance level.
        self.sumsq_sensitivity_df = pd.DataFrame()

        :return: _description_
        :rtype: _type_
        """
        significant_df = self.sumsq_sensitivity_df[
            self.sumsq_sensitivity_df["converged"].ge(self.significance_level)
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

    def read_json(self, file):
        """Read and print a json file.

        :param file: Path to json
        :type file: str
        """
        df = pd.read_json(file, orient="split")
        print(df)

    def plot_rbd_si_indices(self):
        """Calculate RBD FAST Sobol Indices and plot"""
        fig, ax = plt.subplots(1)
        # fig.set_size_inches(18, 5)

        ax.tick_params(labelsize=14)
        sensdf = self.sensitivity_df
        sensdf = sensdf.rename(self.name_dict, axis=0)
        # x-axis
        sensdf.plot(
            kind="barh", y="S1", xerr="S1_conf", ax=ax, align="center", capsize=3
        )
        # y-axis
        ax.set_xlabel("Sensitivity Indices: " + "Fusion power", fontsize=20)
        ax.set_ylabel("PROCESS parameter", fontsize=20)

        # in striped grey : not significant indices
        ax.fill_betweenx(
            y=[-0.5, len(self.sensitivity_df)],
            x1=0,
            x2=self.significance_level,
            color="grey",
            alpha=0.2,
            hatch="//",
            edgecolor="white",
        )
        plt.grid()
        # plt.savefig("plots/sensitivity_fom.svg", bbox_inches="tight")
        plt.show()

    def plot_sumsq_sensitivity(self):
        """Find the input paramters influencing whether PROCESS converges."""
        fig, ax = plt.subplots(1)
        ax.tick_params(labelsize=16)
        sumsqnamedf = self.sumsq_sensitivity_df.rename(self.name_dict, axis=0).clip(
            lower=0.0
        )
        # x-axis
        sumsqnamedf.plot(
            kind="barh",
            y="unconverged",
            ax=ax,
            align="center",
            label="Significance Index",
            capsize=3,
        )

        # y-axis
        ax.set_xlabel("Influence on convergance", fontsize=20)
        ax.set_ylabel("PROCESS parameter", fontsize=20)
        ax.fill_betweenx(
            y=[-0.5, len(self.sumsq_sensitivity_df)],
            x1=0,
            x2=self.significance_level,
            color="grey",
            alpha=0.2,
            hatch="//",
            edgecolor="white",
            label="Region of Insignificance",
        )
        ax.legend(fontsize=12, borderpad=0.01, ncol=1)

        plt.grid()
        plt.savefig("rds_indices.svg", bbox_inches="tight")
        plt.show()

    def convergence_study(self, n, sampled_inputs, process_output):
        """This function is used to calculate RBD FAST sensitivities indices for a subset.
        It draws a random sample, of a given size, from the total dataset. This is used to
        create a study of the convergence of the indices for the input parameters.

        :param n: Number of samples to select
        :type n: int
        :param sampled_inputs: Array of the sampled input parameters
        :type sampled_inputs: numpy.array()
        :param process_output: Array of the figure of merit
        :type process_output: numpy.array()
        :return: Sensitivity Indices dict, length of the subset
        :rtype: dict, int
        """
        subset = np.random.choice(len(process_output), size=n, replace=False)
        self.problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.input_names,
        }
        rbd_results = rbd_fast.analyze(
            self.problem, X=sampled_inputs[subset], Y=process_output[subset]
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
        converged_figure_of_merit_df = self.get_fom_converged_df(figure_of_merit)
        sampled_vars_df = self.converged_sampled_vars_df

        indices_df_list = []
        for n in np.arange(
            start=50, stop=len(converged_figure_of_merit_df) + 1, step=step_size
        ):
            rbd_results, len_subset = self.convergence_study(
                n=n,
                sampled_inputs=sampled_vars_df.to_numpy(),
                process_output=converged_figure_of_merit_df.to_numpy(),
            )
            rbd_results["samples"] = len_subset
            indices_df_list.append(rbd_results.to_df())
        indices_df = pd.concat(indices_df_list)

        return indices_df

    def plot_convergence_study(self, step_size, figure_of_merit):
        """Performs a convergence study and plots the results. Plots confidence levels only if
        final indices are greater than significance level.

        :param step_size: Number of samples to increment by when calculating sensitivity indices
        :type step_size: int
        """
        indices_df = self.find_convergence_indices(step_size, figure_of_merit)
        conv_fig, conv_ax = plt.subplots()
        conv_fig.set_size_inches(10, 6)
        conv_ax.set_ylim(ymin=-0.15, ymax=1)

        for name in self.input_names:
            name_df = indices_df.loc[[name]]
            name_df.plot(
                x="samples",
                y="S1",
                label=self.name_dict[name],
                ax=conv_ax,
                linewidth=3,
            )

            if name_df["S1"].iloc[-1] > self.significance_level:
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
            y2=self.significance_level,
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

    def merge_hdf_files(self):
        """Looks for uncertainty hdf files in the working folder and merges them into a
        single dataframe for analysis.

        :return: Uncertainties DataFrame, Parameters DataFrame
        :rtype: pandas.DataFrame, pandas.Dataframe
        """
        list_uncertainties_dfs = []
        list_params_dfs = []
        for root, dirs, files in os.walk(self.path_in):
            for file in files:
                pos_hdf = root + os.sep + file
                if pos_hdf.endswith(".h5") and "uncertainties_data" in pos_hdf:
                    extra_uncertainties_df = pd.read_hdf(pos_hdf)
                    list_uncertainties_dfs.append(extra_uncertainties_df)
        return pd.concat(list_uncertainties_dfs)

    def create_scatter_plot(
        self,
        axes=None,
        df=None,
        diagonal="density",
        density_kwds=None,
        hist_kwds=None,
        marker="*",
        alpha=0.5,
        color="tab:blue",
        **kwds,
    ):
        """
        Create a scatter plot to visuals inputs against the figure of merit. Used to look for trends in the data at a glance.
        diagonal: either "hist", "kde" or "density"
        See def scatter_matrix in: https://github.com/pandas-dev/pandas/blob/526f40431a51e1b1621c30a4d74df9006e0274b8/pandas/plotting/_matplotlib/misc.py

        """
        range_padding = 0.05
        hist_kwds = hist_kwds or {}
        density_kwds = density_kwds or {}

        ## fix input data
        mask = pd.notna(df)

        boundaries_list = []
        for a in df.columns:
            values = df[a].values[mask[a].values]
            rmin_, rmax_ = np.min(values), np.max(values)
            rdelta_ext = (rmax_ - rmin_) * range_padding / 2.0
            boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

        ## iterate over columns
        for i, a in enumerate(df.columns):
            for j, b in enumerate(df.columns):
                ax = axes[i, j]  ## to abbreviate the code

                if i == j:
                    values = df[a].values[mask[a].values]

                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == "hist":
                        ax.hist(values, color=color, alpha=alpha, **hist_kwds)

                    elif diagonal in ("kde", "density"):
                        from scipy.stats import gaussian_kde

                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), color=color, **density_kwds)

                    ax.set_xlim(boundaries_list[i])
                    # Take the text off inbetween diagonal plots.
                    if i < 4:
                        ax.get_xaxis().set_visible(False)
                else:
                    common = (mask[a] & mask[b]).values
                    h = ax.hist2d(
                        x=df[b][common], y=df[a][common], bins=10, cmap="magma"
                    )
                    if i > j:
                        plt.colorbar(h[3], ax=ax)
                    plt.grid()
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                ax.tick_params(axis="both", which="major", labelsize=12)
                ax.set_xlabel(self.name_dict[b], fontsize=12)
                ax.set_ylabel(self.name_dict[a], fontsize=12)

                if i < j:
                    axes[i, j].set_visible(False)
                if j != 0:
                    ax.yaxis.set_visible(False)

        return

    def plot_scatter_plot(self, plot_unconverged=False):
        """Configures the plot the scatter plot.
        :param plot_unconverged: Plot unconverged runs in a red histogram, defaults to False
        :type plot_unconverged: bool, optional
        """

        figsize = (30, 30)
        figure()
        plt.figure(figsize=figsize)
        axes = {}
        n = self.plot_converged_df.columns.size
        gs = mpl.gridspec.GridSpec(
            n,
            n,
            left=0.12,
            right=0.97,
            bottom=0.12,
            top=0.97,
            wspace=0.025,
            hspace=0.005,
        )
        for i, a in enumerate(self.plot_converged_df.columns):
            for j, b in enumerate(self.plot_converged_df.columns):
                axes[i, j] = plt.subplot(gs[i, j])
        self.create_scatter_plot(
            axes=axes,
            df=self.plot_converged_df,
            diagonal="hist",
            color="#0000ff",
            marker=".",
        )

        if plot_unconverged == True:
            self.create_scatter_plot(
                axes,
                diagonal="hist",
                df=self.plot_unconverged_df,
                color="#ff0000",
                marker=".",
            )
        # plt.savefig("plots/2dhist.svg", bbox_inches="tight")
        plt.show()

    def convergence_regional_sensitivity_analysis(self, variable_names):
        """Regional sensitivity anlysis to find the parameters which influence a converged solution.
        Uses modified RSA technique from SALib."""
        # x = inputs
        # y = figure of merit (sqsumsq)
        figure_of_merit_df = self.uncertainties_df["sqsumsq"]
        sampled_vars_df = self.uncertainties_df[variable_names]
        self.problem = {
            "num_vars": len(sampled_vars_df),
            "names": variable_names,
        }
        rsa_result = rsa.analyze(
            problem=self.problem,
            X=sampled_vars_df.to_numpy(),
            Y=figure_of_merit_df.to_numpy(),
            bins=2,
            target="convergence",
            print_to_console=False,
            seed=1,
            mode="process",
        )
        self.sumsq_sensitivity_df = rsa_result.to_df()
        self.sumsq_sensitivity_df = self.sumsq_sensitivity_df.T
        self.sumsq_sensitivity_df.columns = ["converged", "unconverged"]
        self.sumsq_sensitivity_df = self.sumsq_sensitivity_df.sort_values(
            by="unconverged", ascending=False
        )

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
        self.problem = {
            "num_vars": len(sampled_vars_df),
            "names": variables_to_sample,
        }
        # Regional sensitivity analysis
        rsa_result = rsa.analyze(
            problem=self.problem,
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

    def hdmr_analysis(self):
        """HDMR Analysis - not well explored."""
        fom_df = self.get_fom_converged_df("sqsumsq")
        hdmr_res = hdmr.analyze(
            self.problem, self.converged_sampled_vars_df.to_numpy(), fom_df.to_numpy()
        )
        print(hdmr_res.to_df().to_string())

    def ecdf_plot(self, figure_of_merit):
        """Plot Empirical Cumulative distribution Functions for converged and unconverged samples.
        Additionally, plot the convergence rate and the design point value.

        :param figure_of_merit: Parameter to investigate
        :type figure_of_merit: str
        """
        fig, ax1 = plt.subplots(1)
        plt.rcParams["axes.xmargin"] = 0
        plt.rcParams["axes.ymargin"] = 0
        ax2 = ax1.twinx()
        plt.style.library["tableau-colorblind10"]
        # Arrange the data into df
        converged_fom_df = self.converged_df[figure_of_merit].to_numpy()
        uq_fom_df = self.uncertainties_df[figure_of_merit].to_numpy()
        # Calculate cumulative distribution functions
        ecdf_unconv = sm.distributions.ECDF(uq_fom_df)
        ecdf_conv = sm.distributions.ECDF(converged_fom_df)
        # Bin the data
        x = np.linspace(min(uq_fom_df), max(uq_fom_df))
        ri_t = []
        y_unconv = ecdf_unconv(x)
        y_conv = ecdf_conv(x)
        # Plot the ecdf functions
        ax1.step(x, y_unconv, color="tab:red", label="Unconverged samples")
        ax1.step(x, y_conv, color="tab:blue", label="Converged samples")
        ax1.set_ylabel("Percentage of Samples", fontsize=20)
        ax1.set_xlabel(self.name_dict[figure_of_merit], fontsize=20)
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
        # Finds the edges of bins (must be a better way to do this section)
        h, edges = np.histogram(ri_t, bins=x)
        # Plot convergence rate
        ax1.stairs(ri_t, edges, color="tab:orange", label="Convergence rate")
        ax2.set_ylabel("Convergence Rate", fontsize=20)
        # Plot design point
        ypoint = 0.5
        if figure_of_merit == "kappa":
            ypoint = 1.0
        ### copy curve line y coords and set to a constant
        lines = y_unconv.copy()
        lines[:] = ypoint

        # get intersection
        first_line = LineString(np.column_stack((x, y_unconv)))
        second_line = LineString(np.column_stack((x, lines)))
        intersection = first_line.intersection(second_line)
        ax1.plot(*intersection.xy, "s", color="tab:green", markersize=7)

        # plot hline and vline
        ax1.hlines(
            y=ypoint,
            xmin=min(x),
            xmax=intersection.x,
            label="Design point value",
            clip_on=False,
            color="tab:green",
            linestyles="dashed",
            alpha=0.7,
        )
        ax1.vlines(
            x=intersection.x,
            ymin=0,
            ymax=intersection.y,
            clip_on=True,
            color="tab:green",
            linestyles="dashed",
            alpha=0.7,
        )

        ax1.legend()
        ax1.grid(axis="both")
        # plt.savefig("plots/" + figure_of_merit + "ecdf.svg", bbox_inches="tight")


class Copula:
    """Copula class. Creates a copula from input data and has functions to plot.
    Specify if you want to use bounded or unbounded distributions. Try unbounded
    initially, and with few input variables as there is a wider selection of
    distributions. With higher number of input variables try bounded to contain
    infinites."""

    def __init__(self, input_data, input_names=None, copula_type="unbounded"):
        self.input_data = input_data
        if input_names is not None:
            self.input_data = self.input_data[input_names]
        self.input_names = input_names
        self.variable_names = self.input_data.columns.tolist()
        bounded_univariate = Univariate(
            bounded=True,
        )
        if copula_type == "unbounded":
            self.copula = GaussianMultivariate(distribution=Univariate())
            # vine_type="
        elif copula_type == "bounded":
            self.copula = GaussianMultivariate(distribution=bounded_univariate)
        elif copula_type == "vine":
            self.copula = VineCopula("regular")  # "center","regular","direct"

    def calculate_copula(
        self,
        input_sample_size=None,
        synthetic_sample_size=None,
    ):
        """Randomly sample the input data and fit a copula.

        :param input_sample_size: Number of random samples to take from input data, defaults to 10
        :type input_sample_size: int, optional
        :param synthetic_sample_size: Number of synthetic samples to generate from the copula, defaults to 10
        :type synthetic_sample_size: int, optional
        """
        # Set sample size to length of input data if size is not specified.
        input_sample_size = input_sample_size or len(self.input_data)
        synthetic_sample_size = synthetic_sample_size or len(self.input_data)

        # Sample some random data from the dataset. Now we are just looking at converged samples.
        self.sampled_input_data = self.input_data[self.variable_names].sample(
            n=input_sample_size, random_state=1
        )
        unique_array = unique_cols(self.sampled_input_data)
        self.sampled_input_data = self.sampled_input_data.loc[:, ~unique_array]
        self.copula.fit(self.sampled_input_data)
        # Sample from the copula to create synthetic data.
        self.synthetic_data = self.copula.sample(synthetic_sample_size)

    def copula_data_dict(self):
        """Print some parameters which describe the copula. The correlation matrix indicates
        which parameters are influencing each other. The distribution type tells you which fit
        was applied to each parameter.
        """

        # Output the copula parameters to a dict and print some of them.
        self.copula_dict = self.copula.to_dict()

    def correlation_matrix(self, correlation_matrix):
        """Create a correlation matrix dataframe and affix the variable names as column and index.
        It may not be necessary to keep this as a function.
        :return: correlation_df
        :rtype: pandas.dataframe
        """
        correlation_df = pd.DataFrame(
            correlation_matrix,
            columns=self.input_data.columns,
            index=self.input_data.columns,
        )
        return correlation_df

    def plot_correlation_matrix(self, correlation_matrix):
        """Take the correlation dataframe and plot it in a heatmap style grid.

        :param correlation_matrix: Correlation dataframe, index and columns should be the names of the parameters.
        :type correlation_matrix: Pandas DataFrame
        """
        num_variables = len(correlation_matrix)
        figsize = (min(12, num_variables), min(10, num_variables))

        correlation_matrix_df = pd.DataFrame(
            correlation_matrix,
            columns=self.input_data.columns,
            index=self.input_data.columns,
        )
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix_df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.show()

    def calculate_pdf(self):
        """Returns the joint probability density function for the copula."""

        # Calculate probability density function.
        self.pdf = self.copula.pdf(self.synthetic_data)

    def calculate_cdf(self, data):
        """Returns the joint probability density function for the copula."""

        # Calculate probability density function.
        self.cdf = self.copula.cdf(data)

    def plot_2d(self, variable_x: str, variable_y: str):
        """Plot some 2D scatter plots for inspection

        :param variable_x: Variable for x-axis
        :type variable_x: str
        :param variable_y: Variable for y-axis
        :type variable_y: str
        """
        input_data_2d = self.sampled_input_data[[variable_x, variable_y]].copy()
        synthetic_data_2d = self.synthetic_data[[variable_x, variable_y]].copy()

        scatter_2d(input_data_2d)
        compare_2d(input_data_2d, synthetic_data_2d)

    def plot_3d(self, variable_x: str, variable_y: str, variable_z: str):
        """Plot side-by-side 3D scatter plots for inspection.

        :param variable_x: Variable for x-axis
        :type variable_x: str
        :param variable_y: Variable for y-axis
        :type variable_y: str
        :param variable_z: Variable for z-axis
        :type variable_z: str
        """
        input_data_3d = self.input_data[[variable_x, variable_y, variable_z]].copy()
        synthetic_data_3d = self.synthetic_data[
            [variable_x, variable_y, variable_z]
        ].copy()

        compare_3d(input_data_3d, synthetic_data_3d)

    def create_converged_data(self, variable=None, synthetic_data=None):
        synthetic_data = (
            synthetic_data if synthetic_data is not None else self.synthetic_data
        )
        if variable == None:
            variable = synthetic_data.columns.values
        var_df = synthetic_data[variable]

        return var_df

    def create_cdf_df(self, variable=None, synthetic_data=None):
        synthetic_data = synthetic_data or self.synthetic_data
        if variable == None:
            variable = synthetic_data.columns.values
        var_df = synthetic_data[variable]
        cdf_df = var_df.assign(cdf=self.cdf)
        return cdf_df

    def plot_pdf(self, variable=None, synthetic_data=None):
        """Plot the probability density function for a given variable.

        :param variable: Variable to plot, defaults to str
        :type variable: _type_, optional
        :param synthetic_data: _description_, defaults to None
        :type synthetic_data: _type_, optional
        """
        converged_data = self.create_converged_data(variable, synthetic_data)
        converged_data.sort_values(variable).set_index(variable).plot(kind="area")

    def find_max_pdf(self, variable=None, synthetic_data=None, print_data=False):
        """Find the maximum pdf value and its corresponding data value."""
        converged_data = self.create_converged_data(variable, synthetic_data)
        max_pdf = converged_data.loc[converged_data["pdf"].idxmax()]
        if print_data == True:
            print(max_pdf)
        return max_pdf

    def plot_ecdf_comparison(self, variable):
        """Plot ECDF plot for a given variable."""

        # Prepare some dataframes for comparing ecdf
        input_data_df = self.input_data[variable].to_numpy()
        synthetic_data_df = self.synthetic_data[variable].to_numpy()

        # Calculate cumulative distribution functions to compare plots
        ecdf_input = sm.distributions.ECDF(input_data_df)
        ecdf_synthetic = sm.distributions.ECDF(synthetic_data_df)

        # Bin the data
        x = np.linspace(min(input_data_df), max(input_data_df))
        y_input = ecdf_input(x)
        y_synthetic = ecdf_synthetic(x)

        # Plot the ecdf functions
        fig, ax1 = plt.subplots(1)
        ax1.step(x, y_input, color="tab:blue", label="Input samples")
        ax1.step(x, y_synthetic, color="tab:orange", label="Sythentic samples")
        ax1.set_ylabel("Fraction of Samples", fontsize=20)
        ax1.set_xlabel(variable, fontsize=20)
        ax1.legend(fontsize=12, borderpad=0.01, ncol=1)


class ConfidenceAnalysis:
    """A tool for plotting UQ and Copula data for analysis."""

    def __init__(
        self,
        uq_data,
        input_names,
        num_intervals=2,
        weight_confidence=1.0,
        weight_overlap=0.5,
        custom_data_point=None,
    ):
        """
        Initialize the ConfidenceAnalysis instance.

        Parameters:
        - uq_data: UQ data for analysis.
        - copula: Copula instance for modeling dependencies.
        - num_intervals: Number of intervals for probability calculations.
        - custom_data_point: Custom data point for analysis.
        """

        # 1. Parameter Validation
        if not isinstance(uq_data, UncertaintyData):
            raise ValueError("uq_data must be of type UQDataType.")
        # if not isinstance(copula, Copula):
        #     raise ValueError("copula must be an instance of CopulaType.")

        # 2. Attribute Initialization
        self.num_intervals = num_intervals
        self.weight_confidence = weight_confidence
        self.weight_overlap = weight_overlap
        self.uq_data = uq_data
        self.input_names = input_names
        self.plot_list = []  # list of plots.
        self.variable_data = {}  # list of variable data classes.
        self.plot_height_width = 275  # height and width of plots.
        self.design_values_df = uq_data.estimate_design_values(self.input_names)
        # Here you could switch between real and synthetic data. In general, use the real data
        # but synthetic may be useful if convergence is low.
        self.converged_data = (
            self.uq_data.converged_df
        )  # self.uq_data.converged_df  # self.copula.synthetic_data  #
        self.custom_data_point = custom_data_point
        self.probability_df = pd.DataFrame()

        # 3. Calculate interval probabilities for each variable
        for var in self.input_names:
            best_metric = float("-inf")
            best_config = None
            for num_intervals in range(
                # Square root of number of samples is a general method to estimate the max number of intervals.
                1,
                round(np.sqrt(len(self.uq_data.converged_df))),
            ):
                # I arrived at these values after playing around a bit, more careful examination needed.
                # I think it would be better to check if I can now eliminate the "ghost" interval at the end.
                variable_data = self.calculate_variable_probabilities(
                    variable=var, num_intervals=num_intervals
                )

                # Perform grid search with different configurations
                confidences_grid = variable_data.interval_confidence
                errors_grid = variable_data.interval_confidence_uncertainty

                # Calculate the metric for the current configuration
                current_metric = self.calculate_metric(
                    confidences_grid,
                    errors_grid,
                    self.weight_confidence,
                    self.weight_overlap,
                )
                # print(var, current_metric)
                # Update best configuration if the metric is improved
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_config = (confidences_grid, errors_grid)
            # print(var)
            # print("Best Configuration:")
            # print("Confidences:", best_config[0])
            # print("Errors:", best_config[1])
            # print("Best Metric:", best_metric)
            self.calculate_variable_probabilities(var, len(best_config[0]))
        # 4. Modify data for plotting
        self.plotting_data = self.modify_data(self.variable_data)

    # Define a function to calculate the metric
    def calculate_metric(self, confidences, errors, weight_confidence, weight_overlap):
        metric = sum(
            weight_confidence * confidences
            - weight_overlap * self.overlap(confidences, errors)
        )
        return metric

    # Define a function to calculate the overlap between intervals
    def overlap(self, confidences, errors):
        overlaps = []
        for pair in combinations(range(len(confidences)), 2):
            i, j = pair
            overlap_area = max(
                0,
                min(confidences[i] + errors[i], confidences[j] + errors[j])
                - max(confidences[i] - errors[i], confidences[j] - errors[j]),
            )
            union_area = min(
                confidences[i] + errors[i], confidences[j] + errors[j]
            ) - max(confidences[i] - errors[i], confidences[j] - errors[j])
            overlaps.append(overlap_area / union_area)
            overlaps.append(
                overlap_area / union_area if union_area != 0 else 0
            )  # Handle division by zero

        # Debugging output
        # print(f"Overlap Area: {overlap_area}, Union Area: {union_area}")
        # print(f"Confidences: {confidences}")
        # print(f"Errors: {errors}")
        # print(f"Overlap Area: {overlap_area}")
        # print(f"Union Area: {union_area}")
        return sum(overlaps)

    def _sort_converged_data_by_variable(self, variable: str):
        self.converged_data.sort_values(variable, inplace=True)

    def _get_design_range_and_value(self, variable: str):
        """Look for the minimum/maximum of the design range or the synthetic range.
        Sometimes synthetic can extend beyond the real data. This data is used to create
        grid plots later."""
        # need to rework this slightly for using synthetic data separately todo.
        if variable in self.uq_data.input_names:
            design_range_start, design_range_end, design_value = (
                self.uq_data.uncertainties_df[variable].min(),
                self.uq_data.uncertainties_df[variable].max(),
                self.uq_data.uncertainties_df[variable].mean(),
            )
        else:
            design_range_start, design_range_end, design_value = (
                self.uq_data.converged_df[variable].min(),
                self.uq_data.converged_df[variable].max(),
                self.uq_data.converged_df[variable].mean(),
            )
        return design_range_start, design_range_end, design_value

    def _add_interval_column(self, dataframe, variable: str, design_range_intervals):
        interval_column_name = variable + "_interval"
        dataframe[interval_column_name] = pd.cut(
            dataframe[variable], bins=design_range_intervals, labels=False
        )
        dataframe[interval_column_name] = pd.to_numeric(
            dataframe[interval_column_name], errors="coerce"
        )
        if dataframe[interval_column_name].dtype != "Int64":
            dataframe[interval_column_name] = dataframe[interval_column_name].astype(
                "Int64"
            )

    def _calculate_interval_probability(self, design_range_intervals, variable: str):
        """Sum the pdf values in an interval.

        :param design_range_intervals: _description_
        :type design_range_intervals: _type_
        :param variable: _description_
        :type variable: str
        :return: _description_
        :rtype: _type_
        """
        # Probability is being calculated as number of converged sampples/number of uncertain points.
        converged_intervals = (
            self.converged_data.groupby(variable + "_interval")[variable].count()
            / len(self.uq_data.uncertainties_df)
            # * (1.0 - self.uq_data.failure_probability)
            # Above line modifies the probability to account for the fact that we are looking at converged samples.
        )
        interval_probability = pd.Series(
            0.0, index=pd.RangeIndex(len(design_range_intervals))
        )

        interval_probability.iloc[converged_intervals.index.values.astype(int)] = (
            converged_intervals.values
        )
        # interval_probability /= interval_probability.sum()

        return interval_probability

    def find_description(self, variable_name, trim_after_char=80):
        """Get the variable descriptiuon from the process variable dict.
        Trim patterns and character length.

        :param variable_name: variable to search for
        :type variable_name: str
        :param variable_name: number of characters to trim description after
        :type variable_name: int
        :return: description of variable
        :rtype: str
        """
        variable_dict = python_fortran_dicts.get_dicts()
        description = variable_dict["DICT_DESCRIPTIONS"].get(
            variable_name, "Description not found"
        )

        # Trim the description to the first 100 characters.
        trimmed_description = description[:trim_after_char]
        # Define patterns to remove (with variations).
        patterns_to_remove = [
            r"\(`iteration variable \d+`\)",
            r"\(`calculated \d+`\)",
            r"\(calculated \d+`\)",
        ]  # Add your patterns here
        # Remove specified patterns.
        for pattern in patterns_to_remove:
            trimmed_description = re.sub(pattern, "", trimmed_description)

        return trimmed_description

    def sum_intervals_to_probability(
        self, interval_endpoints, interval_probabilities, desired_probability
    ):
        # Convert pandas dataframe to numpy array
        probabilities = pd.DataFrame(interval_probabilities)
        sorted_probabilities = probabilities.sort_values(by=0, ascending=False)
        # Initialize variables
        cumulative_sum = 0
        included_probabilities = []
        included_index = []

        # Iterate through the sorted DataFrame and add probabilities to the cumulative sum
        for index, row in sorted_probabilities.iterrows():
            probability = row[0]
            included_index.append(included_index)
            cumulative_sum += probability
            included_probabilities.append(probability)
            included_index.append(index)
            # Check if the cumulative sum exceeds the desired probability
            if cumulative_sum >= desired_probability:
                break
        return included_index

    def calculate_variable_probabilities(self, variable: str, num_intervals: int):
        """Use the Joint Probability Function to find the region in uncertain space with the highest rate of convergence.
        The uncertain space is grouped into intervals. The probability density of each point  in the interval is summed.
        This value is normalised, which gives the probability that a converged point has come from this interval.
        """
        # Sort the converged_data by the variable
        self._sort_converged_data_by_variable(variable)
        (
            design_range_start,
            design_range_end,
            design_value,
        ) = self._get_design_range_and_value(variable)

        # Variable intervals
        design_range_intervals = np.linspace(
            design_range_start,
            design_range_end,
            num_intervals,
            endpoint=False,
        )
        design_range_bins = np.append(design_range_intervals, design_range_end)

        if num_intervals == 1:
            self._add_interval_column(
                self.converged_data, variable, [design_range_start, design_range_end]
            )
        elif num_intervals > 1:
            self._add_interval_column(self.converged_data, variable, design_range_bins)
        interval_probability = self._calculate_interval_probability(
            design_range_intervals, variable
        )

        self.converged_data["pdf"] = interval_probability
        # Search var_pdf and map values to intervals, sum pdf over intervals.
        converged_intervals = interval_probability
        # Map values to intervals
        # Map values to intervals with "right" set to False
        int_uncertainties_df = pd.DataFrame()
        int_uncertainties_df["intervals"] = pd.cut(
            self.uq_data.uncertainties_df[variable],
            bins=design_range_bins,
            right=False,
        )
        # This is the number sampled, (includes unconverged samples)
        int_converged_df = pd.DataFrame()
        int_converged_df["intervals"] = pd.cut(
            self.uq_data.converged_df[variable],
            bins=design_range_bins,
            right=False,
        )

        # Count the frequency of values in each interval (converged+unconverged)
        interval_counts = (
            int_uncertainties_df["intervals"].value_counts().sort_index()
        ).tolist()
        conv_interval_counts = (
            int_converged_df["intervals"].value_counts().sort_index()
        ).tolist()
        # Display the results

        interval_probability = pd.Series(
            0.0, index=pd.RangeIndex(len(design_range_bins))
        )
        interval_probability.iloc[converged_intervals.index.values.astype(int)] = (
            converged_intervals.values
        )
        # This is the probability that, of converged samples, it will be found in a given interval.
        interval_probability = interval_probability / interval_probability.sum()
        interval_probability = interval_probability[:-1]
        # Interval width. (The width of intervals in parameter space)#
        if num_intervals == 1:
            interval_width = design_range_end - design_range_start
            design_value_index = 0
            # Interval counts is total sampled intervals (converged+unconverged).
            # Interval probability is
            interval_confidence, interval_con_unc = self._calculate_confidence(
                conv_interval_counts,
                interval_counts,
            )
            design_value_probability = interval_confidence
        elif num_intervals > 1:
            interval_width = design_range_intervals[1] - design_range_intervals[0]
            design_value_index = np.digitize(design_value, design_range_intervals) - 1
            interval_confidence, interval_con_unc = self._calculate_confidence(
                conv_interval_counts,
                interval_counts,
            )
            design_value_probability = interval_confidence[design_value_index]
        # Interval counts is total sampled intervals (converged+unconverged).
        # Interval probability is

        # Store the results for plotting in dataframe
        # Get the input values

        # design_value_probability = interval_confidence[design_value_index]
        # Get the maximum pdf value
        max_confidence = interval_confidence.max()
        max_confidence_index = interval_confidence.argmax()
        max_confidence_design_interval = design_range_intervals[max_confidence_index]
        # Check if a custom_data_point is provided
        if self.custom_data_point is not None:
            if variable in self.custom_data_point.keys():
                custom_data_point_index = (
                    np.digitize(
                        self.custom_data_point[variable], design_range_intervals
                    )
                    - 1
                )
                custom_data_point_probability = interval_confidence[
                    custom_data_point_index
                ]
                custom_data_point_value = self.custom_data_point[variable]
            else:
                custom_data_point_value = None
                custom_data_point_index = None
                custom_data_point_probability = None
        else:
            custom_data_point_value = None
            custom_data_point_index = None
            custom_data_point_probability = None
        # Find variable description in dict
        description = self.find_description(variable)
        # Create variable dataclass and store it
        variable_data = uncertain_variable_data(
            name=variable,
            description=description,
            design_range_start=design_range_start,
            design_range_end=design_range_end,
            design_value=design_value,
            design_range_intervals=design_range_intervals,
            number_of_intervals=len(design_range_intervals),
            interval_probability=interval_probability,
            interval_confidence=interval_confidence,
            interval_confidence_uncertainty=interval_con_unc,
            confidence_sum=sum(interval_confidence),
            interval_sample_counts=interval_counts,
            max_confidence_value=max_confidence_design_interval
            + (0.5 * interval_width),
            max_confidence_lb=max_confidence_design_interval,
            max_confidence_ub=max_confidence_design_interval + interval_width,
            max_confidence=max_confidence,
            design_value_probability=design_value_probability,
            design_value_index=design_value_index,
            max_confidence_index=max_confidence_index,
            interval_width=interval_width,
            custom_data_point=custom_data_point_value,
            custom_data_point_index=custom_data_point_index,
            custom_data_point_probability=custom_data_point_probability,
        )

        self.variable_data[variable] = variable_data

        return variable_data

    def modify_data(self, variable_data):
        """Convert variable data into a format for plotting graphs and tables.

        Parameters:
        - variable_data: List of variable data classes.

        Returns:
        - The results of UQ Data analysis, used for plotting Bokeh Tables and Graphs.
        """
        data = self._convert_variable_data_to_dataframe(variable_data)
        # Joint probability calculations
        self._calculate_joint_confidence(data)

        # Filter dataframes based on design and max probability values
        design_filtered_df = self._filter_dataframe_by_index(
            self.converged_data, data, "design_value_index", self.input_names
        )
        max_confidence_filtered_df = self._filter_dataframe_by_index(
            self.converged_data, data, "max_confidence_index", self.input_names
        )

        def modify_row(row):
            row["custom_data_point_index"] = np.digitize(
                row["custom_data_point_mean"], row["design_range_intervals"]
            )

            if row["custom_data_point_index"] < len(row["interval_confidence"]):
                row["custom_data_point_probability"] = row["interval_confidence"][
                    row["custom_data_point_index"]
                ]
            else:
                # Handle the case where the index is out of bounds
                row["custom_data_point_probability"] = np.nan

            row["custom_data_point_value"] = row["custom_data_point"]

            return row

        # Custom point calculations
        if self.custom_data_point is not None:
            # Filter the data by the given custom points. Note: if a custom point is not
            # given for each input variable the probability can't be directly compared to
            # design or optimised probability (probability will be greater the fewer points
            # provided)
            custom_filtered_df = self._filter_dataframe_by_index(
                self.converged_data,
                data,
                "custom_data_point_index",
                self.custom_data_point.keys(),
            )
            if custom_filtered_df.shape[0] > 0:
                data["custom_data_point_mean"] = custom_filtered_df.mean()
            else:
                data["custom_data_point_mean"] = 0.0
                print("Custom Point is non-convergent. All values are set to zero.")
            data["custom_delta"] = data["custom_data_point_mean"] - data["design_value"]
            data = data.apply(modify_row, axis=1)

            self._calculate_custom_joint_probability(data)

        # Design and max probability means
        data["design_mean"] = (
            design_filtered_df.mean() if design_filtered_df.shape[0] > 0 else 0.0
        )
        data["max_confidence_mean"] = (
            max_confidence_filtered_df.mean()
            if max_confidence_filtered_df.shape[0] > 0
            else 0.0
        )

        # Calculate the delta between design value and max probable value.
        data["optimised_delta"] = data["max_confidence_value"] - data["design_value"]

        return data

    def _convert_variable_data_to_dataframe(self, variable_data):
        """Convert variable data into a DataFrame."""
        data_dict_list = [asdict(variable) for variable in variable_data.values()]
        data = pd.DataFrame(data_dict_list)
        data.set_index("name", inplace=True)
        return data

    def _calculate_joint_confidence(self, data):
        """Calculate the confidence of the original design space, then the confidence of the optimised space."""
        # Joint input probability is the confidence of your original bounds.
        # The confidence of each interval is summed, and divided by the number of intervals. This is averaged for the
        # number of uncertain params you have.
        data["joint_input_probability"] = sum(
            data.loc[self.uq_data.input_names, "confidence_sum"]
            / (
                len(self.input_names)
                * (data.loc[self.uq_data.input_names, "number_of_intervals"])
            )
        )
        # This is the joint maximum confidence. The confidence if you selected the intervals
        # with the highest confidence.
        data["joint_max_confidence"] = data.loc[
            self.uq_data.input_names, "max_confidence"
        ].sum() / (len(self.uq_data.input_names))

    def _calculate_confidence(self, conv_interval_count, interval_sample_counts):
        """Calculate the confidence that an interval will converge. This is defined as
        the ratio of the number of convergent points to sampled points. Currently using
        the generated pdf to estimate convergent points.

        :param interval_probability: probability of a convergent point in the interval
        :type interval_probability: list
        :param num_converged: number of converged points in run
        :type num_converged: int
        :param interval_sample_counts: number of times the interval was sampled by MC.
        :type interval_sample_counts: list
        :return: interval confidence
        :rtype: list
        """
        # Interval probability is probability of converged samples being in an interval (as a proportion of converged samples)
        interval_confidence = (np.array(conv_interval_count)) / (
            np.array(interval_sample_counts)
        )
        delta_conv_interval_count = np.sqrt(conv_interval_count)
        delta_interval_sample_counts = np.sqrt(interval_sample_counts)
        # Suppress this for now, find a solution later :~)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Your division operation here
            delta_c_over_c = np.sqrt(
                (delta_conv_interval_count / conv_interval_count) ** 2
                + (delta_interval_sample_counts / interval_sample_counts) ** 2
            )
        delta_confidence = interval_confidence * delta_c_over_c

        for i in range(len(interval_confidence - 1)):
            if interval_confidence[i] == float("inf") or interval_confidence[
                i
            ] == float("-inf"):
                interval_confidence[i] = 0.0
        return interval_confidence, delta_confidence

    def _filter_dataframe_by_index(
        self, dataframe, variabledata, index_column, columns_to_filter
    ):
        """Filter dataframe based on the given index_column. Data is variable dataframe."""
        return filter_dataframe_by_columns_and_values(
            dataframe, variabledata, columns_to_filter, index_column, self.uq_data.itv
        )

    def _calculate_custom_joint_probability(self, data):
        """Calculate joint probability for custom data points."""
        custom_significant_conv_data = data.loc[
            self.uq_data.significant_conv_vars, "custom_data_point_probability"
        ]
        data["joint_custom_probability"] = custom_significant_conv_data.product()

    def create_plot(self, uncertain_variable):
        """Create some plots to show how the probability of convergence is deconvolved into the uncertain space for each variable.
        This function plots bar charts which show the probability of convergence for intervals in uncertain variable space.
        The red line plots the design point values. The green line finds the bar with the maximum probability (crude estimate).
        The probabilities of the bars for each variable sum to 1, as does the entire uncertain space (we are only looking at converged).
        The probabilities of the variable should be multiplied to find the joint probability.
        The grey grid is used to plot the uncertain space.

        :param uncertain_variable: Uncertain variable to plot
        :type uncertain_variable: str
        :return: A bokeh figure
        :rtype: Bokeh Figure
        """
        # Uncertain Variable Data (uncertain_variable_data)
        uncertain_variable_data = self.plotting_data.loc[uncertain_variable]
        DESIGN_RANGE_START_FACTOR = 0.95
        DESIGN_RANGE_END_FACTOR = 1.05

        # Plot input variable lines
        design_value_box = BoxAnnotation(
            left=uncertain_variable_data.design_value,
            right=uncertain_variable_data.design_value,
            top=uncertain_variable_data.max_confidence,
            bottom=0.0,
            line_color="red",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
            name="Design Point Value",
        )

        #
        # design_value_probability_box = BoxAnnotation(
        #     left=uncertain_variable_data.design_range_start,
        #     right=uncertain_variable_data.design_value,
        #     top=uncertain_variable_data.design_value_probability,
        #     bottom=uncertain_variable_data.design_value_probability,
        #     line_color="red",
        #     line_width=2,
        #     line_alpha=1.0,
        #     line_dash="dashed",
        # )
        sample_space = HBar(
            y=uncertain_variable_data.max_confidence * 0.5,
            right=uncertain_variable_data.design_range_end,
            left=uncertain_variable_data.design_range_start,
            height=uncertain_variable_data.max_confidence,
            fill_alpha=0.1,
            fill_color="grey",
            line_alpha=0.2,
            hatch_pattern="dot",
            hatch_scale=3,
            hatch_alpha=0.15,
        )
        # Plot max pdf lines
        max_var_box = BoxAnnotation(
            left=uncertain_variable_data.max_confidence_value,
            right=uncertain_variable_data.max_confidence_value,
            top=uncertain_variable_data.max_confidence,
            bottom=0.0,
            line_color="limegreen",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
            name="Max PDF value",
        )
        max_confidence_box = BoxAnnotation(
            left=uncertain_variable_data.design_range_start,
            right=uncertain_variable_data.max_confidence_value,
            top=uncertain_variable_data.max_confidence,
            bottom=uncertain_variable_data.max_confidence,
            line_color="limegreen",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
        )

        p = figure(
            x_range=(
                uncertain_variable_data.design_range_start * DESIGN_RANGE_START_FACTOR,
                uncertain_variable_data.design_range_end * DESIGN_RANGE_END_FACTOR,
            ),
            height=self.plot_height_width,
            width=self.plot_height_width,
            title="Convergance Probability",
        )
        # Check if the varialbe is in the name dictionary and replace the name with description.
        uncertain_variable_name = self.uq_data.name_dict.get(
            uncertain_variable_data.name, uncertain_variable_data.name
        )

        p.xaxis.axis_label = uncertain_variable_name
        p.yaxis.axis_label = "Confidence"
        p.add_glyph(sample_space)
        vert_bar_plot = p.vbar(
            x=uncertain_variable_data.design_range_intervals
            + 0.5 * uncertain_variable_data.interval_width,
            top=uncertain_variable_data.interval_confidence,
            width=uncertain_variable_data.interval_width,
            fill_color="cornflowerblue",
            alpha=0.5,
        )

        errsource = ColumnDataSource(
            data=dict(
                base=uncertain_variable_data.design_range_intervals
                + 0.5 * uncertain_variable_data.interval_width,
                lower=uncertain_variable_data.interval_confidence
                + uncertain_variable_data.interval_confidence_uncertainty,
                upper=uncertain_variable_data.interval_confidence
                - uncertain_variable_data.interval_confidence_uncertainty,
            )
        )

        whisker = Whisker(
            source=errsource,
            base="base",
            upper="upper",
            lower="lower",
        )

        p.add_layout(design_value_box)
        # p.add_layout(design_value_probability_box)
        p.add_layout(max_var_box)
        # p.add_layout(max_confidence_box)
        p.add_layout(whisker)
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Probability", "@top{0.000}"),
                    (uncertain_variable_data.name, "@x{0.000}"),
                ],
                renderers=[vert_bar_plot],
            )
        )
        p.legend
        p.title.text_font = "helvetica"  # Set the font family

        # Save the plot as svg
        p.output_backend = "svg"
        export_svgs(p, filename="plots/" + uncertain_variable_data.name + "_plot.svg")

        return p

    def create_graph_grid(self, variables):
        """Create a grid of graphs which plot the probability
        intervals for each input variable.

        :param variables: Copula variables.
        :type variables: List (str)
        :return: Grid of bokeh graphs
        :rtype: bokeh.gridplot
        """
        for var in variables:
            p = self.create_plot(var)
            input
            self.plot_list.append(p)

        num_plots = len(self.plot_list)
        num_columns = 3
        # Create a grid layout dynamically
        probability_vbar_grid = gridplot(
            [
                self.plot_list[i : i + num_columns]
                for i in range(0, num_plots, num_columns)
            ]
        )
        return probability_vbar_grid

    def create_datatable(self, variables):
        """Create a datatable which summarises findings from the copula."

        :param variables: Copula variables
        :type variables: List(str)
        """
        general_formatter = HTMLTemplateFormatter(
            template='<span style="color: black;"><%- value %></span>'
        )

        columns = [
            TableColumn(
                field="name",
                title="Variable",
                formatter=general_formatter,
            ),
            TableColumn(
                field="design_value",
                title="Design Value",
                formatter=general_formatter,
            ),
            TableColumn(
                field="max_confidence_value",
                title="Optimised Value",
                formatter=general_formatter,
            ),
            TableColumn(
                field="optimised_delta",
                title="Optimised Delta",
                formatter=general_formatter,
            ),
            TableColumn(
                field="max_confidence_lb",
                title="Optimised LB",
                formatter=general_formatter,
            ),
            TableColumn(
                field="max_confidence_ub",
                title="Optimised UB",
                formatter=general_formatter,
            ),
            TableColumn(
                field="interval_width",
                title="Interval Width",
                formatter=general_formatter,
            ),
            TableColumn(
                field="joint_input_probability",
                title="Design Confidence",
                formatter=general_formatter,
            ),
            TableColumn(
                field="joint_max_confidence",
                title="Optimised Confidence",
                formatter=general_formatter,
            ),
            # This prediction isn't very good, so I will comment it out for now.
        ]
        if self.custom_data_point is not None:
            columns.insert(
                4,
                TableColumn(
                    field="custom_data_point_mean",
                    title="Custom Point",
                    formatter=general_formatter,
                ),
            )
            columns.insert(
                5,
                TableColumn(
                    field="custom_delta",
                    title="Custom Delta",
                    formatter=general_formatter,
                ),
            )
            columns.append(
                TableColumn(
                    field="joint_custom_probability",
                    title="Custom Probability",
                    formatter=general_formatter,
                )
            )

        plotting_data = self.plotting_data
        plotting_data = plotting_data.map(format_number)
        source = ColumnDataSource(plotting_data)
        data_table = DataTable(
            source=source,
            columns=columns,
            autosize_mode="force_fit",
            width=1000,
            height_policy="auto",
        )

        return data_table

    def correlation_network(self, correlation_matrix, variables=None, threshold=0.1):
        """
        Create a correlation network based on the given correlation matrix.

        Parameters:
        - correlation_matrix (pd.DataFrame): The correlation matrix.
        - variables (list): List of variables to include. If None, include all variables.
        - threshold (float): Threshold for including edges based on correlation.

        Returns:
        - nx.Graph: The correlation network as a NetworkX graph.
        """
        # Create a graph from the correlation matrix
        G = nx.Graph()
        # Add nodes
        G.add_nodes_from(correlation_matrix)

        # Add edges with weights (correlations)
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # To avoid duplicate edges
                    correlation_value = correlation_matrix.iloc[i, j]
                    if abs(correlation_value) >= threshold:
                        # Include only specified variables and their neighbors
                        if (
                            variables is None
                            or (col1 in variables)
                            or (col2 in variables)
                        ):
                            G.add_edge(col1, col2, weight=correlation_value)
        # Determine nodes to include in the plot
        if variables is None:
            # If no specific variables are specified, include all nodes
            included_nodes = G.nodes
            title_variables = "All Variables"
        else:
            # Include specified variables and their neighbors
            included_nodes = set(variables)
            for variable in variables:
                included_nodes.update(G.neighbors(variable))
            title_variables = ", ".join(variables)

        subgraph = G.subgraph(included_nodes)
        self.included_nodes = list(included_nodes)

        return subgraph

    def plot_network(self, networkx, fig_height=800, fig_width=800):
        """Create a Bokeh network plot. Clickable nodes.

        :param networkx: networkx data
        :type networkx: networkx
        :param correlation_matrix: correlation matrix produced by the copula
        :type correlation_matrix: dict
        """
        variable_data = self._convert_variable_data_to_dataframe(self.variable_data)
        variable_data = variable_data[variable_data.index.isin(self.included_nodes)]
        variable_data = variable_data.map(format_number)
        mapping = dict((n, i) for i, n in enumerate(networkx.nodes))
        G = nx.relabel_nodes(networkx, mapping)
        graph_renderer = from_networkx(G, nx.spring_layout, k=0.9, center=(0, 0))
        graph_renderer.node_renderer.data_source.data["index"] = list(range(len(G)))
        graph_renderer.node_renderer.data_source.data["name"] = (
            variable_data.index.tolist()
        )

        graph_renderer.node_renderer.data_source.data["description"] = variable_data[
            "description"
        ]
        graph_renderer.node_renderer.data_source.data["design_range_start"] = (
            variable_data["design_range_start"]
        )
        graph_renderer.node_renderer.data_source.data["design_range_end"] = (
            variable_data["design_range_end"]
        )
        graph_renderer.node_renderer.data_source.data["max_confidence_value"] = (
            variable_data["max_confidence_value"]
        )

        graph_renderer.node_renderer.data_source.data["node_color"] = [
            PuBuGn5[3] if n in self.uq_data.itv else PuBuGn5[2]
            for n in graph_renderer.node_renderer.data_source.data["name"]
        ]
        plot = figure(
            width=fig_width,
            height=fig_width,
            x_range=Range1d(-1.1, 1.1),
            y_range=Range1d(-1.1, 1.1),
        )
        plot.title.text = "PROCESS UQ Network"
        tooltips = [
            ("Name", "@name"),
            ("Description", "@description"),
            ("Optimised value", "@max_confidence_value"),
            ("Range start", "@design_range_start"),
            ("Range end", "@design_range_end"),
        ]
        plot.add_tools(
            HoverTool(tooltips=tooltips),
            TapTool(),
            BoxSelectTool(),
            ResetTool(),
        )
        circle_size = 70

        graph_renderer.node_renderer.glyph = Circle(
            size=circle_size,
            fill_color="node_color",
        )
        graph_renderer.node_renderer.selection_glyph = Circle(
            size=circle_size, fill_color=PuBuGn5[1]
        )
        graph_renderer.node_renderer.nonselection_glyph = Circle(
            size=circle_size, fill_color="node_color"
        )
        graph_renderer.node_renderer.hover_glyph = Circle(
            size=circle_size, fill_color=PuBuGn5[1]
        )

        # Create labels for nodes and add them to the plot
        x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
        source = ColumnDataSource(
            {
                "x": x,
                "y": y,
                "names": graph_renderer.node_renderer.data_source.data["name"],
            }
        )
        labels = LabelSet(
            x="x",
            y="y",
            text="names",
            level="glyph",
            text_color="black",
            x_offset=0.0,
            y_offset=0.0,
            source=source,
            text_align="center",
            text_baseline="middle",
        )
        plot.add_layout(labels)
        # Draw edge labels separately ... Todo: find a nice way to present this data.
        edge_labels = {
            (i, j): f"{networkx[i][j]['weight']:.2f}" for i, j in networkx.edges()
        }
        # Add gray lines when nothing is highlighted.
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="#CCCCCC", line_alpha=0.8, line_width=5
        )
        edge_data = graph_renderer.edge_renderer.data_source.data
        # Add red lines for positive correlation, blue for negative. Visible on highlight.
        edge_data["line_color"] = [
            PuBu5[0] if w < -0.0 else PuRd5[0] for w in edge_data["weight"]
        ]
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color="line_color", line_width=5
        )
        graph_renderer.edge_renderer.hover_glyph = MultiLine(
            line_color="line_color", line_width=5
        )

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesAndLinkedEdges()

        plot.renderers.append(graph_renderer)
        # Create a custom legend
        negative_circle = plot.circle(
            x=0,
            y=0,
            size=0,
            name="Negative",
            fill_color=PuBu5[0],
            legend_label="Negative",
        )
        negative_circle = plot.circle(
            x=0,
            y=0,
            size=0,
            name="Negative",
            fill_color=PuRd5[0],
            legend_label="Positive",
        )
        plot.legend.title = "Correlation"
        plot.renderers.append(labels)

        show(plot)


@dataclass
class uncertain_variable_data:
    """Used to contain data about uncertain variables, used for plotting." """

    name: str
    description: str
    design_range_start: float
    design_range_end: float
    design_value: float
    design_range_intervals: np.array
    number_of_intervals: int
    interval_probability: np.array
    interval_confidence: pd.DataFrame
    interval_confidence_uncertainty: pd.DataFrame
    confidence_sum: float
    interval_sample_counts: list
    max_confidence_value: float
    max_confidence_lb: float
    max_confidence_ub: float
    max_confidence: float
    design_value_probability: float
    design_value_index: int
    max_confidence_index: int
    interval_width: float
    custom_data_point: float
    custom_data_point_index: float
    custom_data_point_probability: float


def format_number(
    val,
    threshold_small=0.1,
    threshold_large=1e5,
    format_small="{:.3e}",
    format_large="{:.2f}",
):
    """
    Format a number based on thresholds. If the absolute value of the number is less than
    threshold_small, it will be formatted using `format_small`. If it's greater than or
    equal to threshold_small and less than threshold_large, it will be formatted using
    `format_large`. Otherwise, it will be formatted using `format_small`.
    """
    if isinstance(val, (int, float)):
        if abs(val) < threshold_small:
            return format_small.format(val)
        elif abs(val) >= threshold_small and abs(val) < threshold_large:
            return format_large.format(val)
        else:
            return format_small.format(val)
    else:
        return val


def filter_dataframe_by_columns_and_values(
    dataframe,
    interval_data,
    columns_to_filter,
    value_to_filter,
    iteration_variables,
):
    """
    Filter a DataFrame based on specified columns and corresponding values.

    Parameters:
        - dataframe (pd.DataFrame): The DataFrame to filter.
        - columns_to_filter (list of str): A list of column names to filter.
        - values_to_match (list): A list of values to match for each corresponding column.

    Returns:
        - pd.DataFrame: A new DataFrame containing rows where the specified columns match the corresponding values.
    """
    filter_condition = pd.Series(True, index=dataframe.index)
    for column in columns_to_filter:
        if column in iteration_variables:
            pass
        else:
            filter_condition &= (
                dataframe[column + "_interval"]
                == interval_data.loc[column][value_to_filter]
            )
    filtered_df = dataframe[filter_condition]

    return filtered_df


def filter_dataframe_between_ranges(dataframe, column_name, lower_bound, upper_bound):
    """
    Filter a DataFrame based on a specified column and range values.

    Parameters:
    - dataframe: pandas DataFrame
    - column_name: str, the column on which filtering is to be applied
    - lower_bound: numeric, the lower bound of the range (inclusive)
    - upper_bound: numeric, the upper bound of the range (inclusive)

    Returns:
    - filtered_dataframe: pandas DataFrame, the filtered DataFrame
    """
    mask = (dataframe[column_name] >= lower_bound) & (
        dataframe[column_name] <= upper_bound
    )
    filtered_dataframe = dataframe[mask]
    return filtered_dataframe


def unique_cols(df):
    """Find columns which are not all the same value.

    :param df: DataFrame in question
    :type df: pandas.DataFrame
    :return: DataFrame containing with no constant columns
    :rtype: pandas.DataFrame
    """
    a = df.to_numpy()
    return (a[0] == a).all(0)
