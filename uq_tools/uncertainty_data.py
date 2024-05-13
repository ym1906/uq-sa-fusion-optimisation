import pandas as pd
import os
from SALib.analyze import rbd_fast, rsa
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import LineString
from bokeh.plotting import figure


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
        self.uncertainties_df = merge_hdf_files(self.path_in)
        # Remove columns which have the same value in every row.
        self.unique_array = unique_cols(self.uncertainties_df)
        self.uncertainties_df = self.uncertainties_df.loc[:, ~self.unique_array]
        self.uncertainties_df["sqsumsq"] = np.log(self.uncertainties_df["sqsumsq"])
        self.sampled_variables = sampled_variables
        self.number_sampled_vars = len(self.sampled_variables)
        self.problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.sampled_variables,
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
        self.converged_sampled_vars_df = self.converged_df[self.sampled_variables]
        self.unconverged_sampled_vars_df = self.unconverged_df[self.sampled_variables]
        self.significance_level = 0.05  # Significance level is used for plotting, indicates level below which we consider
        # index values insignificant, 0.05 by default - subjective.
        self.number_of_converged_runs = len(self.converged_df.index)
        self.number_of_unconverged_runs = len(self.unconverged_df)

    def estimate_design_values(uq_dataframe, variables):
        """Find the mean values of sampled parameters as a guess of input initial value
        (assumes uniform distribution)"""
        design_values_df = uq_dataframe[variables]
        mean_values_df = design_values_df.mean(axis=0)

        return mean_values_df

    def configure_data_for_plotting(self):
        """This function sorts the UQ data into dataframes for plotting with the hdf_to_scatter tool."""
        if len(self.sampled_vars_to_plot) == 0:
            print("Plotting all sampled parameters")
            self.plot_names.extend(self.sampled_variables)
            self.plot_converged_df = self.converged_df[self.plot_names]
            self.plot_unconverged_df = self.unconverged_df[self.plot_names]
        else:
            print("Ploting user named parameters")
            self.plot_names.extend(self.sampled_vars_to_plot)

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

    def calculate_sensitivity(
        self,
        figure_of_merit,
    ):
        """Calculate the sensitivity indices for a set of converged UQ runs.
        Uses the Salib rbd_fast analysis method.
        """
        converged_figure_of_merit_df = self.get_fom_converged_df(figure_of_merit)
        sampled_vars_df = self.converged_sampled_vars_df
        problem = {
            "num_vars": self.number_sampled_vars,
            "names": self.sampled_variables,
        }
        sirbd_fast = rbd_fast.analyze(
            problem,
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

    def plot_rbd_si_indices(self):
        """Calculate RBD FAST Sobol Indices and plot"""
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

    def plot_sumsq_sensitivity(self, export_svg=False, svg_path=None):
        """Find the input paramters influencing whether PROCESS converges."""
        fig, ax = plt.subplots(1)
        ax.tick_params(labelsize=16)
        sumsqnamedf = self.sumsq_sensitivity_df.rename(
            process_variable_dict, axis=0
        ).clip(lower=0.0)
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
        if export_svg:
            if svg_path:
                # Use the provided export path
                filename = os.path.join(svg_path, "rds_indices.svg")
            else:
                # Use the default directory of the script
                filename = os.path.join(os.path.dirname(__file__), "rds_indices.svg")
            plt.savefig(filename, bbox_inches="tight")
        plt.show()

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
        converged_figure_of_merit_df = self.get_fom_converged_df(figure_of_merit)
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

        for name in self.sampled_variables:
            name_df = indices_df.loc[[name]]
            name_df.plot(
                x="samples",
                y="S1",
                label=process_variable_dict[name],
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
        See def scatter_matrix in:
        https://github.com/pandas-dev/pandas/blob/526f40431a51e1b1621c30a4d74df9006e0274b8/pandas/plotting/_matplotlib/misc.py
        """
        range_padding = 0.05
        hist_kwds = hist_kwds or {}
        density_kwds = density_kwds or {}

        # fix input data
        mask = pd.notna(df)

        boundaries_list = []
        for a in df.columns:
            values = df[a].values[mask[a].values]
            rmin_, rmax_ = np.min(values), np.max(values)
            rdelta_ext = (rmax_ - rmin_) * range_padding / 2.0
            boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

        # Iterate over columns.
        for i, a in enumerate(df.columns):
            for j, b in enumerate(df.columns):
                ax = axes[i, j]  # to abbreviate the code

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
                ax.set_xlabel(process_variable_dict[b], fontsize=12)
                ax.set_ylabel(process_variable_dict[a], fontsize=12)

                if i < j:
                    axes[i, j].set_visible(False)
                if j != 0:
                    ax.yaxis.set_visible(False)

        return

    def plot_scatter_plot(self, plot_unconverged=False):
        """Configures the plot the scatter plot.
        :param plot_unconverged: Plot unconverged runs in a red histogram.
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

        if plot_unconverged is True:
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
        problem = {
            "num_vars": len(sampled_vars_df),
            "names": variable_names,
        }
        rsa_result = rsa.analyze(
            problem=problem,
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
        ax1.set_xlabel(process_variable_dict[figure_of_merit], fontsize=20)
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
        # copy curve line y coords and set to a constant
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
    "rmajor": "major radius (m)",
    "tmargtf": "Temperature margin TF coil",
    "dene": "dene",
    "ohcth": "ohcth",
    "beta": "beta",
    "betalim": "betalim",
    "n_cycle_min": "Minimum number of allowable stress cycles",
}


def read_json(file):
    """Read and print a json file.

    :param file: Path to json
    :type file: str
    """
    df = pd.read_json(file, orient="split")
    print(df)
