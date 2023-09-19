import pandas as pd
import argparse
import json
import os
from dataclasses import dataclass, asdict
from SALib.analyze import rbd_fast, hdmr, rsa
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import figure
from shapely.geometry import LineString

import math
from copulas.visualization import (
    scatter_2d,
    compare_2d,
    compare_3d,
)
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate import (
    ParametricType,
    Univariate,
    GaussianUnivariate,
    GaussianKDE,
    BetaUnivariate,
    GammaUnivariate,
)
from bokeh.palettes import Viridis256
from bokeh.io import curdoc
from bokeh.plotting import figure, show, row
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    LabelSet,
    HoverTool,
    DataTable,
    TableColumn,
    Div,
    NumberFormatter,
    Column,
    CustomJS,
    Select,
    BoxAnnotation,
    VArea,
    HBar,
    Legend,
    LegendItem,
    FuncTickFormatter,
    HTMLTemplateFormatter,
)


def parse_args(args):
    """Parse supplied arguments.

    :param args: arguments to parse
    :type args: list, None
    :return: parsed arguments
    :rtype: Namespace
    """
    parser = argparse.ArgumentParser(
        description="Calulate sensitivity analysis indices for Monte Carlo runs of PROCESS."
    )

    parser.add_argument(
        "-iv",
        "--input_sampled_vars",
        default="param_values.h5",
        help="input hdf5 file with sample points (default=param_values.h5)",
    )

    parser.add_argument(
        "-uf",
        "--uq_folder",
        default="uq_data_folder",
        help="Folder containing all UQ files",
    )

    parser.add_argument(
        "-f",
        "--configfile",
        default="config_evaluate_uncertainties.json",
        help="configuration file",
    )

    parser.add_argument(
        "-iu",
        "--input_uncertainty",
        default="uncertainties_data.h5",
        help="input hdf5 file with uncertainty data (default=param_values.h5)",
    )

    parser.add_argument(
        "-fom",
        "--figure_of_merit",
        default="rmajor",
        help="figure of merit, must be pulled from input uncertainty (default=param_values.h5)",
    )

    return parser.parse_args(args)


class UncertaintyData:
    """Collects and analyses the output from the evaluate_uncertainties.py tool. Supply an instance with the input
    folder for the run data. The tool looks for hdf files containing uncertainty data, merges them, and has functions
    to clean, analyse, and plot the data."""

    def __init__(self, path_to_uq_data_folder, figure_of_merit):
        self.path_in = path_to_uq_data_folder
        self.figure_of_merit = figure_of_merit

        self.sample_vars_h5_path = "param_values.h5"
        self.uncertainties_h5_path = "uncertainties_data.h5"
        self.uncertainties_df, self.sampled_vars_df = self.merge_hdf_files()
        self.uncertainties_df = self.uncertainties_df.drop(
            columns=[
                "procver",
                "date",
                "time",
                "username",
                "runtitle",
                "tagno",
                "branch_name",
                "commsg",
                "fileprefix",
                "tauelaw",
                # "radius_of_plasma-facing_side_of_inner_leg_should_be_[m]",
                "error_id",
            ]
        )
        # Remove columns which have the same value in every row.
        self.unique_array = unique_cols(self.uncertainties_df)
        self.uncertainties_df = self.uncertainties_df.loc[:, ~self.unique_array]
        self.uncertainties_df["sqsumsq"] = np.log(self.uncertainties_df["sqsumsq"])
        # self.uncertainties_df = self.uncertainties_df.head(100)
        self.input_names = self.sampled_vars_df.columns.tolist()
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
            "gapoh",
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
            "gapds",
            "plhthresh",
            "tmargtf",
            "tmargoh",
            "oh_steel_frac",
            "pdivt",
            "powfmw",
        ]
        # self.input_names.extend(itv)
        self.sampled_vars_to_plot = []
        self.plot_names = []
        self.number_sampled_vars = len(self.input_names)
        self.problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.input_names,
        }
        self.converged_df = self.uncertainties_df[self.uncertainties_df["ifail"] == 1.0]
        self.unconverged_df = self.uncertainties_df[
            self.uncertainties_df["ifail"] != 1.0
        ]
        self.converged_sampled_vars_df = self.converged_df[self.input_names]
        self.unconverged_sampled_vars_df = self.unconverged_df[self.input_names]
        self.unconverged_fom_df = self.uncertainties_df[self.figure_of_merit]
        self.plot_converged_df = pd.DataFrame()
        self.plot_unconverged_df = pd.DataFrame()
        self.sensitivity_df = pd.DataFrame()
        self.sumsq_sensitivity_df = pd.DataFrame()
        self.converged_sampled = pd.DataFrame()
        # Significance level is used for plotting, indicates level below which we consider
        # index values insignificant, 0.05 by default - subjective.
        self.significance_level = 0.05
        self.significant_conv_vars = []

        self.reliability_index = 0.0
        self.number_of_converged_runs = len(self.converged_df.index)
        self.number_of_unconverged_runs = len(self.unconverged_df)
        self.item_with_sensitivity = []
        self.sens_dict = {}
        self.list_new_dfs = []

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
            "wallmw": "wallmw",
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

    def find_most_sensitive_interaction(self, parameters_to_search):
        """Perform a sensitivity analysis on a list of variables, store if influential."""
        for item in parameters_to_search:
            self.figure_of_merit = item
            # The error here is caused by paramters which have the same value in every run.
            # For example q0 which is 1.0 for every run.
            self.calculate_sensitivity(item)
            if self.sensitivity_df.isnull().values.any() & self.sensitivity_df.empty:
                pass
            else:
                new_df = self.sensitivity_df[
                    self.sensitivity_df["S1"] > self.significance_level
                ]
                if new_df.empty:
                    pass
                else:
                    if len(new_df) == self.number_sampled_vars:
                        pass
                    else:
                        new_df.insert(2, "variable", item)
                        self.item_with_sensitivity.append(item)
                        self.sens_dict[item] = new_df.to_dict()
                        self.list_new_dfs.append(new_df)
        print(self.item_with_sensitivity)
        mc_run_df = pd.concat(self.list_new_dfs)
        print(mc_run_df)
        for name in self.input_names:
            if name in mc_run_df.index:
                name_df = mc_run_df.loc[[name]]
                print("Number influenced by ", name, len(name_df))
                print(name_df)
                name_df.to_json(
                    self.path_in
                    + name
                    + "_influence_"
                    + str(self.figure_of_merit)
                    + ".json",
                    orient="split",
                    compression="infer",
                    index="true",
                )
        with open("result_05si.json", "w") as fp:
            json.dump(self.sens_dict, fp)

    def calculate_reliability(self):
        """Calculate the Reliability Index, defined as number of converged runs divided by
        the number of failed runs.
        """
        self.reliability_index = round(
            (len(self.converged_df.index)) / len((self.uncertainties_df.index)), 2
        )

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
        sumsqnamedf = self.sumsq_sensitivity_df  # .rename(self.name_dict, axis=0)
        # x-axis
        sumsqnamedf.plot(
            kind="barh",
            y="unconverged",
            ax=ax,
            align="center",
            label="Converged",
            capsize=3,
        )
        # y-axis
        ax.set_xlabel("Influence on convergance", fontsize=20)
        ax.set_ylabel("PROCESS parameter", fontsize=20)
        ax.fill_betweenx(
            y=[-0.5, len(self.sensitivity_df)],
            x1=0,
            x2=self.significance_level,
            color="grey",
            alpha=0.2,
            hatch="//",
            edgecolor="white",
            label="Not significant",
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
                if pos_hdf.endswith(".h5") and "param_values" in pos_hdf:
                    extra_params_df = pd.read_hdf(pos_hdf)
                    list_params_dfs.append(extra_params_df)
        return pd.concat(list_uncertainties_dfs), pd.concat(list_params_dfs)

    def create_scatter_plot(
        self,
        axes=None,
        df=None,
        diagonal="hist",
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
                        x=df[b][common], y=df[a][common], bins=3, cmap="magma"
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
        figure(figsize=figsize)
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

    def parallel_plot(self):
        """Parallel plots are used to display paths to convergence. Lines track from one parameter to the next.
        This allows you to look for paths through the parameters which lead to convergence.
        """
        minValue = -24.0
        maxValue = -8.0
        # vtp=variables to plot
        vtp = [
            "etath",
            "kappa",
        ]
        input_sample = ot.Sample.BuildFromDataFrame(self.uncertainties_df[vtp])
        fom_sample_np = self.uncertainties_df["sqsumsq"].to_numpy()
        print(fom_sample_np)
        fom_sample = ot.Sample([[v] for v in fom_sample_np])
        quantileScale = False
        graphCobweb = ot.VisualTest.DrawParallelCoordinates(
            input_sample,
            fom_sample,
            minValue,
            maxValue,
            "red",
            quantileScale,
        )
        graphCobweb.setLegendPosition("bottomright")
        view = viewer.View(graphCobweb, figure_kw={"figsize": (32, 18)})


class Copula:
    """Copula class. Creates a copula from input data and has functions to plot."""

    def __init__(self, input_data, input_names=None):
        self.input_data = input_data
        if input_names is not None:
            self.input_data = self.input_data[input_names]
        self.input_names = input_names
        self.sampled_input_data = pd.DataFrame()
        self.variable_names = self.input_data.columns.tolist()
        self.pdf = []
        self.cdf = []
        self.synthetic_data = pd.DataFrame()
        univariate = Univariate(parametric=ParametricType.PARAMETRIC)

        self.copula = GaussianMultivariate()  # distribution=univariate
        self.copula_dict = {}

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
        self.copula.fit(self.sampled_input_data)
        # Sample from the copula to create synthetic data.
        self.synthetic_data = self.copula.sample(synthetic_sample_size)

    def print_copula_data(self):
        """Print some parameters which describe the copula. The correlation matrix indicates
        which parameters are influencing each other. The distribution type tells you which fit
        was applied to each parameter.
        """

        # Output the copula parameters to a dict and print some of them.
        self.copula_dict = self.copula.to_dict()
        print(self.copula_dict)
        print("Correlation matrix", self.copula_dict["correlation"])

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

    def create_pdf_df(self, variable=None, synthetic_data=None):
        synthetic_data = synthetic_data or self.synthetic_data
        if variable == None:
            variable = synthetic_data.columns.values
        var_df = synthetic_data[variable]
        pdf_df = var_df.assign(pdf=self.pdf)
        return pdf_df

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
        pdf_df = self.create_pdf_df(variable, synthetic_data)
        pdf_df.sort_values(variable).set_index(variable).plot(kind="area")

    def find_max_pdf(self, variable=None, synthetic_data=None, print_data=False):
        """Find the maximum pdf value and its corresponding data value."""
        pdf_df = self.create_pdf_df(variable, synthetic_data)
        max_pdf = pdf_df.loc[pdf_df["pdf"].idxmax()]
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


class InteractivePlot:
    def __init__(
        self, UncertaintyData, Copula, num_intervals=10, custom_data_point=None
    ):
        self.num_intervals = num_intervals
        self.copula = Copula
        self.uq_data = UncertaintyData
        self.pdf_df = Copula.create_pdf_df(self.copula.input_names)
        self.p_list = []
        self.variable_data = {}
        self.height_width = 350
        self.design_values_df = UncertaintyData.estimate_design_values(
            self.copula.input_names
        )
        self.custom_data_point = custom_data_point
        self.probability_df = pd.DataFrame()
        for var in self.copula.input_names:
            self.sort_data(var)

        self.plotting_data = self.modify_data(self.variable_data)

    def sort_data(self, variable):
        """Use the Joint Probability Function to find the region in uncertain space with the highest rate of convergence.
        The uncertain space is grouped into intervals. The probability density of each point  in the interval is summed.
        This value is normalised, which gives the probability that a converged point has come from this interval.
        """
        # Sort the pdf_df by the variable
        self.pdf_df.sort_values(self.copula.input_names)
        # Get the start and end point of the uncertain space. Treat significant convergence variables differently to
        # itertation variables.
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
        # Calculate probabilities. Divide uncertain space into intervals, sum probability in the intervals.
        num_intervals = self.num_intervals
        # Variable intervals
        design_range_intervals = np.linspace(
            design_range_start,
            design_range_end,
            num_intervals,
        )
        self.pdf_df[variable + "_interval"] = pd.cut(
            self.pdf_df[variable], bins=design_range_intervals, labels=False
        )
        # Sometimes intervals weren't integers
        self.pdf_df[variable + "_interval"] = pd.to_numeric(
            self.pdf_df[variable + "_interval"], errors="coerce"
        )
        # Check if the dtype is compatible with 'Int64' and cast if needed
        if self.pdf_df[variable + "_interval"].dtype != "Int64":
            self.pdf_df[variable + "_interval"] = self.pdf_df[
                variable + "_interval"
            ].astype("Int64")
        # Search var_pdf and map values to intervals, sum pdf over intervals.
        converged_intervals = self.pdf_df.groupby(variable + "_interval")["pdf"].sum()
        interval_probability = pd.Series(
            0.0, index=pd.RangeIndex(len(design_range_intervals))
        )
        interval_probability.iloc[
            converged_intervals.index.values.astype(int)
        ] = converged_intervals.values
        interval_probability = interval_probability / interval_probability.sum()
        # Store the results for plotting in dataframe
        # Get the input values
        design_value_index = np.digitize(design_value, design_range_intervals)
        design_value_probability = interval_probability[design_value_index]
        # Get the maximum pdf value
        max_probability = interval_probability.max()
        max_probability_index = interval_probability.argmax()
        max_probability_design_interval = design_range_intervals[max_probability_index]
        # Interval width
        interval_width = (design_range_intervals[-1] - design_range_intervals[0]) / len(
            design_range_intervals
        )
        # Check if a custom_data_point is provided
        if self.custom_data_point is not None:
            if variable in self.custom_data_point.keys():
                custom_data_point_index = np.digitize(
                    self.custom_data_point[variable], design_range_intervals
                )
                custom_data_point_probability = interval_probability[
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
        # Create variable dataclass and store it
        variable_data = uncertain_variable_data(
            name=variable,
            design_range_start=design_range_start,
            design_range_end=design_range_end,
            design_value=design_value,
            design_range_intervals=design_range_intervals,
            interval_probability=interval_probability,
            max_probability_value=max_probability_design_interval,
            max_probability=max_probability,
            design_value_probability=design_value_probability,
            design_value_index=design_value_index,
            max_probability_index=max_probability_index,
            interval_width=interval_width,
            custom_data_point=custom_data_point_value,
            custom_data_point_index=custom_data_point_index,
            custom_data_point_probability=custom_data_point_probability,
        )

        self.variable_data[variable] = variable_data

    def modify_data(self, variable_data):
        """Convert variable data into a format for plotting graphs and tables. We want to look at how PROCESS
        uses iteration variables to respond to changes in the uncertain values. This function looks for the range
        of select variables that correspond to the intervals containing the design and optimal point respectively.
        The mean is take of this range as an approximation for evaluating the delta between input and optimum.

        :param variable_data: List of variable dataclasses
        :type variable_data: List
        :return: The results of UQ Data analysis, used for plotting Bokeh Tables and Graphs
        :rtype: pandas.DataFrame
        """
        # Convert variable data into a DataFrame
        data_dict_list = [asdict(variable) for variable in variable_data.values()]
        data = pd.DataFrame(data_dict_list)
        data.set_index("name", inplace=True)
        # Only use the significant convergence parameters for evaluating probability.
        input_significant_conv_data = data.loc[
            self.uq_data.significant_conv_vars, "design_value_probability"
        ]
        max_significant_conv_data = data.loc[
            self.uq_data.significant_conv_vars, "max_probability"
        ]
        custom_significant_conv_data = data.loc[
            self.uq_data.significant_conv_vars, "custom_data_point_probability"
        ]
        data["joint_input_probability"] = input_significant_conv_data.product()
        data["joint_max_probability"] = max_significant_conv_data.product()
        data["joint_custom_probability"] = custom_significant_conv_data.product()
        # Search for the range of variables of interest which correspond to the design value and max probability value.
        columns_to_filter = self.copula.input_names
        design_filtered_df = filter_dataframe_by_columns_and_values(
            self.pdf_df, data, columns_to_filter, "design_value_index", self.uq_data.itv
        )
        max_probability_filtered_df = filter_dataframe_by_columns_and_values(
            self.pdf_df,
            data,
            columns_to_filter,
            "max_probability_index",
            self.uq_data.itv,
        )
        custom_filtered_df = filter_dataframe_by_columns_and_values(
            self.pdf_df,
            data,
            columns_to_filter,
            "custom_data_point_index",
            self.uq_data.itv,
        )
        # Check if these dataframes are empty, this may be redundant now.
        if design_filtered_df.shape[0] > 0:
            data["design_mean"] = design_filtered_df.mean().round(3)
        else:
            data["design_mean"] = "Non-convergent"

        if max_probability_filtered_df.shape[0] > 0:
            data["max_probability_mean"] = max_probability_filtered_df.mean().round(3)
        else:
            data["max_probability_mean"] = "Non-convergent"
        if custom_filtered_df.shape[0] > 0:
            data["custom_data_point_mean"] = custom_filtered_df.mean().round(3)
        else:
            data["custom_data_point_mean"] = "Non-convergent"
        # Calculate the delta between design value and max probable value.
        data["optimised_delta"] = data["max_probability_mean"] - data["design_value"]
        if self.custom_data_point is not None:
            data["custom_delta"] = data["custom_data_point_mean"] - data["design_value"]

        return data

    def create_plot(self, uncertain_variable):
        """Create some plots to show how the probability of convergence is deconvolved into the uncertain space for each variable.
        This function plots bar charts which show the probability of convergence for intervals in uncertain variable space.
        The red line plots the design point values. The green line finds the bar with the maximum probability (crude estimate).
        The probabilities of the bars for each variable sum to 1, as does the entire uncertain space (we are only looking at converged).
        The probabilities of the variable should be multiplied to find the joint probability.
        The grey grid is used to plot the uncertain space.

        :param uncertain_variable_data: DataClass containing relevant data for each uncertain variable
        :type uncertain_variable_data: DataClass
        :return: A bokeh figure
        :rtype: Bokeh Figure
        """
        # Uncertain Variable Data (uvd)
        uvd = self.plotting_data.loc[uncertain_variable]

        # Plot input variable lines
        design_value_box = BoxAnnotation(
            left=uvd.design_value,
            right=uvd.design_value,
            top=uvd.design_value_probability,
            bottom=0.0,
            line_color="red",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
            name="Design Point Value",
        )
        design_value_probability_box = BoxAnnotation(
            left=uvd.design_range_start,
            right=uvd.design_value,
            top=uvd.design_value_probability,
            bottom=uvd.design_value_probability,
            line_color="red",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
        )
        sample_space = HBar(
            y=uvd.max_probability * 0.5,
            right=uvd.design_range_end,
            left=uvd.design_range_start,
            height=uvd.max_probability,
            fill_alpha=0.1,
            fill_color="grey",
            line_alpha=0.2,
            hatch_pattern="dot",
            hatch_scale=3,
            hatch_alpha=0.15,
        )
        # Plot max pdf lines
        max_var_box = BoxAnnotation(
            left=uvd.max_probability_value,
            right=uvd.max_probability_value,
            top=uvd.max_probability,
            bottom=0.0,
            line_color="limegreen",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
            name="Max PDF value",
        )
        max_probability_box = BoxAnnotation(
            left=uvd.design_range_start,
            right=uvd.max_probability_value,
            top=uvd.max_probability,
            bottom=uvd.max_probability,
            line_color="limegreen",
            line_width=2,
            line_alpha=1.0,
            line_dash="dashed",
        )

        p = figure(
            x_range=(uvd.design_range_start * 0.95, uvd.design_range_end * 1.05),
            height=self.height_width,
            width=self.height_width,
            title="Convergance Probability",
        )
        p.xaxis.axis_label = uvd.name
        p.yaxis.axis_label = "Normalised Probability"
        p.add_glyph(sample_space)
        vbar = p.vbar(
            x=uvd.design_range_intervals,
            top=uvd.interval_probability,
            width=(uvd.design_range_intervals[-1] - uvd.design_range_intervals[0])
            / len(uvd.design_range_intervals),
            fill_color="cornflowerblue",
        )
        p.add_layout(design_value_box)
        p.add_layout(design_value_probability_box)
        p.add_layout(max_var_box)
        p.add_layout(max_probability_box)
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Probability", "@top{0.000}"),
                    (uvd.name, "@x{0.000}"),
                ],
                renderers=[vbar],
            )
        )

        return p

    def create_layout(self, variables, plot_graph=True, plot_table=True):
        """Create a bokeh layout with graphs to epxlain uncertain variables. Create a datatable which
        summarises findings"

        :param variables: Variables which form the Copula
        :type variables: List
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
                field="max_probability_mean",
                title="Optimised Value",
                formatter=general_formatter,
            ),
            TableColumn(
                field="optimised_delta",
                title="Optimised Delta",
                formatter=general_formatter,
            ),
            TableColumn(
                field="interval_width",
                title="Interval Width",
                formatter=general_formatter,
            ),
            TableColumn(
                field="joint_input_probability",
                title="Design Probability",
                formatter=general_formatter,
            ),
            TableColumn(
                field="joint_max_probability",
                title="Optimised Probability",
                formatter=general_formatter,
            ),
        ]
        if self.custom_data_point is not None:
            columns.append(
                TableColumn(
                    field="custom_data_point_mean",
                    title="Custom Point",
                    formatter=general_formatter,
                )
            )
            columns.append(
                TableColumn(
                    field="custom_delta",
                    title="Custom Delta",
                    formatter=general_formatter,
                )
            )
            columns.append(
                TableColumn(
                    field="joint_custom_probability",
                    title="Custom Probability",
                    formatter=general_formatter,
                )
            )
        for var in variables:
            p = self.create_plot(var)
            input
            self.p_list.append(p)

        num_plots = len(self.p_list)
        num_columns = 3
        num_rows = math.ceil(num_plots / num_columns)
        # Create a grid layout dynamically
        grid = gridplot(
            [self.p_list[i : i + num_columns] for i in range(0, num_plots, num_columns)]
        )
        if plot_graph == True:
            show(grid)
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
        # data_table.sizing_mode = "stretch_both"
        if plot_table == True:
            show(data_table)


@dataclass
class uncertain_variable_data:
    """Used to contain data about uncertain variables, used for plotting." """

    name: str
    design_range_start: float
    design_range_end: float
    design_value: float
    design_range_intervals: np.array
    interval_probability: np.array
    max_probability_value: float
    max_probability: float
    design_value_probability: float
    design_value_index: int
    max_probability_index: int
    interval_width: float
    custom_data_point: float
    custom_data_point_index: float
    custom_data_point_probability: float


def format_number(
    val,
    threshold_small=0.001,
    threshold_large=1e5,
    format_small="{:.2e}",
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


def main(args=None):
    args = parse_args(args)
    uq_data = UncertaintyData(args.uq_folder, args.figure_of_merit)
    uq_data.calculate_sensitivity()
    uq_data.calculate_reliability()
    uq_data.configure_data_for_plotting()
    uq_data.plot_scatter_plot(plot_unconverged=False)


def unique_cols(df):
    """Find columns which are not all the same value.

    :param df: DataFrame in question
    :type df: pandas.DataFrame
    :return: DataFrame containing with no constant columns
    :rtype: pandas.DataFrame
    """
    a = df.to_numpy()
    return (a[0] == a).all(0)


if __name__ == "__main__":
    main()
