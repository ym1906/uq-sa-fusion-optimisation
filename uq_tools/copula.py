import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import numpy as np


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


def unique_cols(df):
    """Find columns which are not all the same value.

    :param df: DataFrame in question
    :type df: pandas.DataFrame
    :return: DataFrame containing with no constant columns
    :rtype: pandas.DataFrame
    """
    a = df.to_numpy()
    return (a[0] == a).all(0)
