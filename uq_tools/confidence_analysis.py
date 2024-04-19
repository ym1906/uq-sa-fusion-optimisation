import os
import inspect
import pandas as pd
import re
from dataclasses import dataclass, asdict
import numpy as np
import networkx as nx
import uncertainty_data as UncertaintyData
from pylab import figure
from bokeh.palettes import (
    PuBu5,
    PuRd5,
    PuBuGn5,
)
import python_fortran_dicts as python_fortran_dicts
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
from uncertainty_data import UncertaintyData


class ConfidenceAnalysis:
    """A tool for plotting UQ and Copula data for analysis. This class performs an interval
    analysis. It can calcualte the `confidence` of each interval for each parameter
    and plot this in a grid. Then you can also plot a data table to find the most confident
    interval.
    """

    def __init__(
        self,
        uq_data,
        input_names,
        number_intervals=2,
        weight_confidence=1.0,
        weight_overlap=0.5,
        custom_data_point=None,
    ):
        """
        Initialize the ConfidenceAnalysis instance.

        Parameters:
        - uq_data: UQ data for analysis.
        - copula: Copula instance for modeling dependencies.
        - number_intervals: Number of intervals for probability calculations.
        - custom_data_point: Custom data point for analysis.
        """

        # 1. Parameter Validation
        if not isinstance(uq_data, UncertaintyData):
            raise ValueError("uq_data must be of type UQDataType.")

        # 2. Attribute Initialization
        self.number_intervals = number_intervals
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

    def run(self):
        """Calculate the confidence intervals, perform optimisation, and prepare the data for plotting."""
        # 3. Calculate interval probabilities for each variable
        for variable in self.input_names:
            best_metric = float("-inf")
            best_config = None
            max_number_intervals = round(np.sqrt(len(self.uq_data.converged_df)))
            # Square root of number of samples is a general method to estimate the max number of intervals.
            for number_intervals in range(
                1,
                max_number_intervals,
            ):
                variable_data = self.calculate_variable_probabilities(
                    variable=variable, number_intervals=number_intervals
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

                # Update best configuration if the metric is improved
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_config = (confidences_grid, errors_grid)

            self.calculate_variable_probabilities(variable, len(best_config[0]))
        # 4. Modify data for plotting
        self.plotting_data = self.modify_data(self.variable_data)

    def calculate_metric(self, confidences, errors, weight_confidence, weight_overlap):
        """Calculate a metric to evaluate the number of intervals.

        :param confidences: Confidence values for each intervals.
        :type confidences: np.array
        :param errors: Error values for the confidence of each array.
        :type errors: np.array
        :param weight_confidence: Weighting factor which favours the highest confidence.
        :type weight_confidence: float
        :param weight_overlap: Weighting factor which favours no error overlap.
        :type weight_overlap: flaot
        :return: The value of the metric.
        :rtype: float
        """
        metric = sum(
            weight_confidence * confidences
            - weight_overlap * self.overlap(confidences, errors)
        )
        return metric

    # Define a function to calculate the overlap between intervals
    def overlap(self, confidences, errors):
        """Calculate the overlap between intervals (union of intervals)

        :param confidences: Confidence of each interval.
        :type confidences: np.array
        :param errors: Errors on confidence of each interval.
        :type errors: np.array
        :return: Sum of overlaps between intervals.
        :rtype: float
        """
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

        return sum(overlaps)

    def _sort_converged_data_by_variable(self, variable: str):
        """Sort the a dataframe by a given variable.

        :param variable: Variable name.
        :type variable: str
        """
        self.converged_data = self.converged_data.sort_values(variable)

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
        """Calculate the probability of convegence in an interval.

        :param design_range_intervals: Sampled intervals
        :type design_range_intervals: _type_
        :param variable: Variable of interest.
        :type variable: str
        :return: Probability of convergence in each sampled interval.
        :rtype: _type_
        """
        # Probability is being calculated as number of converged samples/number of uncertain points.
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
        """Get the variable description from the process variable dict.
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

    def calculate_variable_probabilities(self, variable: str, number_intervals: int):
        """Finds the intervals in uncertain space with the highest rate of convergence.
        :param variable: Variable of interest
        :type variable: str
        :param number_intervals: Number of intervals to create
        :type number_intervals: int
        """
        # Sort the converged_data by the variable.
        self._sort_converged_data_by_variable(variable)
        (
            design_range_start,
            design_range_end,
            design_value,
        ) = self._get_design_range_and_value(variable)

        # Create intervals over the entire sampled range.
        design_range_intervals = np.linspace(
            design_range_start,
            design_range_end,
            number_intervals,
            endpoint=False,
        )
        design_range_bins = np.append(design_range_intervals, design_range_end)

        if number_intervals == 1:
            self._add_interval_column(
                self.converged_data, variable, [design_range_start, design_range_end]
            )
        elif number_intervals > 1:
            self._add_interval_column(self.converged_data, variable, design_range_bins)
        interval_probability = self._calculate_interval_probability(
            design_range_intervals, variable
        )

        self.converged_data["pdf"] = interval_probability
        # Search var_pdf and map values to intervals, sum pdf over intervals.
        converged_intervals = interval_probability
        # Map values to intervals
        # Map values to intervals with "right" set to False
        interval_uncertainties_df = pd.DataFrame()
        interval_uncertainties_df["intervals"] = pd.cut(
            self.uq_data.uncertainties_df[variable],
            bins=design_range_bins,
            right=False,
        )
        # This is the number sampled, (includes unconverged samples)
        interval_converged_df = pd.DataFrame()
        interval_converged_df["intervals"] = pd.cut(
            self.uq_data.converged_df[variable],
            bins=design_range_bins,
            right=False,
        )

        # Count the frequency of values in each interval (converged+unconverged)
        interval_counts = (
            interval_uncertainties_df["intervals"].value_counts().sort_index()
        ).tolist()
        converged_interval_counts = (
            interval_converged_df["intervals"].value_counts().sort_index()
        ).tolist()

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
        if number_intervals == 1:
            interval_width = design_range_end - design_range_start
            design_value_index = 0
            # Interval counts is total sampled intervals (converged+unconverged).
            # Interval probability is
            interval_confidence, interval_con_unc = self._calculate_confidence(
                converged_interval_counts,
                interval_counts,
            )
            design_value_probability = interval_confidence
        elif number_intervals > 1:
            interval_width = design_range_intervals[1] - design_range_intervals[0]
            design_value_index = np.digitize(design_value, design_range_intervals) - 1
            interval_confidence, interval_con_unc = self._calculate_confidence(
                converged_interval_counts,
                interval_counts,
            )
            design_value_probability = interval_confidence[design_value_index]

        # Interval counts is total sampled intervals (converged+unconverged).
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
                custom_data_point_interval_index = (
                    np.digitize(
                        self.custom_data_point[variable], design_range_intervals
                    )
                    - 1
                )
                custom_data_point_interval_probability = interval_confidence[
                    custom_data_point_interval_index
                ]
                custom_data_point_interval_value = self.custom_data_point[variable]
            else:
                custom_data_point_interval_value = None
                custom_data_point_interval_index = None
                custom_data_point_interval_probability = None
        else:
            custom_data_point_interval_value = None
            custom_data_point_interval_index = None
            custom_data_point_interval_probability = None
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
            custom_data_point=custom_data_point_interval_value,
            custom_data_point_interval_index=custom_data_point_interval_index,
            custom_data_point_interval_probability=custom_data_point_interval_probability,
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
        self._calculate_jointerval_confidence(data)

        # Filter dataframes based on design and max probability values
        design_filtered_df = self._filter_dataframe_by_index(
            self.converged_data, data, "design_value_index", self.input_names
        )
        max_confidence_filtered_df = self._filter_dataframe_by_index(
            self.converged_data, data, "max_confidence_index", self.input_names
        )

        def modify_row(row):
            row["custom_data_point_interval_index"] = np.digitize(
                row["custom_data_point_interval_mean"], row["design_range_intervals"]
            )

            if row["custom_data_point_interval_index"] < len(
                row["interval_confidence"]
            ):
                row["custom_data_point_interval_probability"] = row[
                    "interval_confidence"
                ][row["custom_data_point_interval_index"]]
            else:
                # Handle the case where the index is out of bounds
                row["custom_data_point_interval_probability"] = np.nan

            row["custom_data_point_interval_value"] = row["custom_data_point"]

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
                "custom_data_point_interval_index",
                self.custom_data_point.keys(),
            )
            if custom_filtered_df.shape[0] > 0:
                data["custom_data_point_interval_mean"] = custom_filtered_df.mean()
            else:
                data["custom_data_point_interval_mean"] = 0.0
                print("Custom Point is non-convergent. All values are set to zero.")
            data["custom_delta"] = (
                data["custom_data_point_interval_mean"] - data["design_value"]
            )
            data = data.apply(modify_row, axis=1)

            self._calculate_custom_jointerval_probability(data)

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

    def _calculate_jointerval_confidence(self, data):
        """Calculate the confidence of the original design space, then the confidence of the optimised space."""
        # Joint input probability is the confidence of your original bounds.
        # The confidence of each interval is summed, and divided by the number of intervals. This is averaged for the
        # number of uncertain params you have.
        data["jointerval_input_probability"] = sum(
            data.loc[self.uq_data.input_names, "confidence_sum"]
            / (
                len(self.input_names)
                * (data.loc[self.uq_data.input_names, "number_of_intervals"])
            )
        )
        # This is the joint maximum confidence. The confidence if you selected the intervals
        # with the highest confidence.
        data["jointerval_max_confidence"] = data.loc[
            self.uq_data.input_names, "max_confidence"
        ].sum() / (len(self.uq_data.input_names))

    def _calculate_confidence(self, converged_interval_count, interval_sample_counts):
        """Calculate the confidence that an interval will converge.
        This is defined as the ratio of the number of convergent points to sampled points.
        Currently using the generated pdf to estimate convergent points.

        :param interval_probability: probability of a convergent point in the interval
        :type interval_probability: list
        :param number_converged: number of converged points in run
        :type number_converged: int
        :param interval_sample_counts: number of times the interval was sampled by MC.
        :type interval_sample_counts: list
        :return: interval confidence
        :rtype: list
        """
        # Convert input lists to numpy arrays
        converged_interval_count = np.array(converged_interval_count)
        interval_sample_counts = np.array(interval_sample_counts)

        # Calculate interval confidence
        with np.errstate(divide="ignore", invalid="ignore"):
            interval_confidence = np.divide(
                converged_interval_count, interval_sample_counts
            )

        # Handle cases where division by zero or NaN values occurred
        interval_confidence[np.isnan(interval_confidence)] = 0  # Replace NaN with 0
        interval_confidence[np.isinf(interval_confidence)] = 0  # Replace inf with 0
        delta_converged_interval_count = np.sqrt(converged_interval_count)
        delta_interval_sample_counts = np.sqrt(interval_sample_counts)
        # Suppress this for now, find a solution later :~)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Your division operation here
            delta_c_over_c = np.sqrt(
                (delta_converged_interval_count / converged_interval_count) ** 2
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

    def _calculate_custom_jointerval_probability(self, data):
        """Calculate joint probability for custom data points."""
        custom_significant_converged_data = data.loc[
            self.uq_data.significant_converged_vars,
            "custom_data_point_interval_probability",
        ]
        data["jointerval_custom_probability"] = (
            custom_significant_converged_data.product()
        )

    def create_plot(self, uncertain_variable, export_svg=False):
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

        # # Plot input variable lines
        sample_space = HBar(
            y=uncertain_variable_data.max_confidence * 0.5,
            right=uncertain_variable_data.design_range_end,
            left=uncertain_variable_data.design_range_start,
            height=uncertain_variable_data.max_confidence
            * 1.1,  # Adding in some extra hehight to make plots look better
            fill_alpha=0.1,
            fill_color="grey",
            line_alpha=0.2,
            hatch_pattern="dot",
            hatch_scale=3,
            hatch_alpha=0.15,
        )
        # Plot max confidence lines
        max_var_box = BoxAnnotation(
            left=uncertain_variable_data.max_confidence_lb,
            right=uncertain_variable_data.max_confidence_ub,
            top=uncertain_variable_data.max_confidence,
            bottom=0.0,
            # line_color="limegreen",
            fill_color="limegreen",
            hatch_pattern="diagonal_cross",
            hatch_scale=10,
            hatch_alpha=0.150,
            # line_width=2,
            # line_alpha=1.0,
            # line_dash="dashed",
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

        p.xaxis.axis_label = "Range: " + uncertain_variable_name
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
        # Whisker is the error bars
        whisker = Whisker(
            source=errsource,
            base="base",
            upper="upper",
            lower="lower",
        )
        whisker.upper_head.size = 20
        whisker.lower_head.size = 20
        # p.add_layout(design_value_box)
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

        if export_svg:
            # Save the plot as SVG
            filename = os.path.join(
                os.path.dirname(__file__), f"{uncertain_variable}_plot.svg"
            )
            p.output_backend = "svg"
            export_svgs(p, filename=filename)

        return p

    def create_graph_grid(self, variables, export_svg=False):
        """Create a grid of graphs which plot the probability
        intervals for each input variable.

        :param variables: Copula variables.
        :type variables: List (str)
        :return: Grid of bokeh graphs
        :rtype: bokeh.gridplot
        """
        for variable in variables:
            p = self.create_plot(variable, export_svg)
            input
            self.plot_list.append(p)

        number_plots = len(self.plot_list)
        number_columns = 3
        # Create a grid layout dynamically
        probability_vbar_grid = gridplot(
            [
                self.plot_list[i : i + number_columns]
                for i in range(0, number_plots, number_columns)
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
                field="jointerval_input_probability",
                title="Design Confidence",
                formatter=general_formatter,
            ),
            TableColumn(
                field="jointerval_max_confidence",
                title="Optimised Confidence",
                formatter=general_formatter,
            ),
            TableColumn(
                field="design_range_start",
                title="Range start",
                formatter=general_formatter,
            ),
            TableColumn(
                field="design_range_end",
                title="Range End",
                formatter=general_formatter,
            ),
            # This prediction isn't very good, so I will comment it out for now.
        ]
        if self.custom_data_point is not None:
            columns.insert(
                4,
                TableColumn(
                    field="custom_data_point_interval_mean",
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
                    field="jointerval_custom_probability",
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

    def plot_network(self, networkx, fig_width_height=800):
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
            width=fig_width_height,
            height=fig_width_height,
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
    custom_data_point_interval_index: float
    custom_data_point_interval_probability: float


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
