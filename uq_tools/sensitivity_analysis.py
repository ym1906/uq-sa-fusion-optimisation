import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import export_png, export_svg
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.palettes import Spectral11
from statsmodels.api import distributions as sm_distributions
import seaborn as sns
from uncertainty_data import process_variable_dict
from matplotlib.colors import to_rgb, to_hex
from tabulate import tabulate  # If you want a neat table format
from typing import List, Dict, Any, Tuple
import json
import re


def initialize_intervals(
    x_min: float, x_max: float, num_intervals: int
) -> List[Dict[str, Any]]:
    interval_width = (x_max - x_min) / num_intervals
    return [
        {
            "x_min": x_min + i * interval_width,
            "x_max": x_min + (i + 1) * interval_width,
            "points": [],
            "num_samples": 0,  # Track the number of samples in each interval
        }
        for i in range(num_intervals)
    ]


def assign_points_to_intervals(
    x: List[float], y: List[float], intervals: List[Dict[str, Any]]
) -> None:
    for xi, yi in zip(x, y):
        for interval in intervals:
            if interval["x_min"] <= xi < interval["x_max"]:
                interval["points"].append(yi)
                interval["num_samples"] += 1  # Increment the sample count
                break


def merge_intervals(
    intervals: List[Dict[str, Any]], min_points: int, x_min: float, x_max: float
) -> List[Dict[str, Any]]:
    merged_intervals = []
    temp_points = []
    temp_num_samples = 0  # Track the number of samples in the merged intervals
    for interval in intervals:
        temp_points.extend(interval["points"])
        temp_num_samples += interval["num_samples"]
        if len(temp_points) >= min_points:
            y_min, y_max = min(temp_points), max(temp_points)
            merged_intervals.append(
                {
                    "x_min": (
                        interval["x_min"]
                        if not merged_intervals
                        else merged_intervals[-1]["x_max"]
                    ),
                    "x_max": interval["x_max"],
                    "y_min": y_min,
                    "y_max": y_max,
                    "num_samples": temp_num_samples,  # Store the number of samples in the merged interval
                }
            )
            temp_points = []  # Reset temp_points after merging
            temp_num_samples = 0  # Reset sample count after merging
    # Handle leftover points
    if temp_points:
        if merged_intervals:
            merged_intervals[-1]["y_max"] = max(
                merged_intervals[-1]["y_max"], max(temp_points)
            )
            merged_intervals[-1][
                "num_samples"
            ] += temp_num_samples  # Add leftover samples to the last merged interval
        else:
            y_min, y_max = min(temp_points), max(temp_points)
            merged_intervals.append(
                {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "num_samples": temp_num_samples,
                }
            )
    return merged_intervals


def calculate_sensitivity_index(
    merged_intervals: List[Dict[str, Any]], total_area: float
) -> float:
    scatter_area = sum(
        (interval["x_max"] - interval["x_min"])
        * (interval["y_max"] - interval["y_min"])
        for interval in merged_intervals
    )
    return 1 - (scatter_area / total_area)


def create_scatter_plot(
    p: figure,
    data: Dict[str, List[float]],
    x_variable: str,
    y_variable: str,
    name,
    weight: float,
    color_mapper: LinearColorMapper,
) -> None:
    scatter_source = ColumnDataSource(
        data=dict(
            x=data[x_variable],
            y=data[y_variable],
            weight=[weight] * len(data[x_variable]),
        )
    )
    p.scatter(
        x="x",
        y="y",
        source=scatter_source,
        size=8,
        color={"field": "weight", "transform": color_mapper},
        legend_label=name,
    )


def add_intervals_to_plot(
    p: figure, merged_intervals: List[Dict[str, Any]], color: str
) -> None:
    for interval in merged_intervals:
        p.varea(
            x=[
                interval["x_min"],
                interval["x_min"],
                interval["x_max"],
                interval["x_max"],
            ],
            y1=[
                interval["y_min"],
                interval["y_max"],
                interval["y_max"],
                interval["y_min"],
            ],
            fill_alpha=0.2,
            fill_color=color,
        )
        p.patch(
            x=[
                interval["x_min"],
                interval["x_min"],
                interval["x_max"],
                interval["x_max"],
                interval["x_min"],
            ],
            y=[
                interval["y_min"],
                interval["y_max"],
                interval["y_max"],
                interval["y_min"],
                interval["y_min"],
            ],
            line_color="black",
            line_width=1,
            fill_alpha=0,
        )


def sensitivity_analysis(
    data_dict: List[Dict[str, Any]],
    x_variable: str,
    y_variable: str,
    num_intervals: int = 10,
    min_points: int = 5,
    export_image: bool = False,
    export_path: str = ".",
    plot_width: int = 800,
    plot_height: int = 600,
    hover_tool: bool = False,
    show_graph: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform sensitivity analysis, calculate sensitivity indices, and merge intervals with insufficient points.

    Parameters:
    - data_dict: List of dictionaries containing data and metadata.
    - x_variable: The variable to be used on the x-axis.
    - y_variable: The variable to be used on the y-axis.
    - num_intervals: Number of intervals for sensitivity analysis.
    - min_points: Minimum number of points per interval.
    - export_image: Whether to export the graph as an image.
    - export_path: Path to save the exported image.
    - plot_width: Width of the plot.
    - plot_height: Height of the plot.
    - hover_tool: Whether to add a hover tool to the plot.
    - show_graph: Whether to display the graph.

    Returns:
    - List of dictionaries containing sensitivity results and merged intervals.
    """
    p = figure(
        width=plot_width,
        height=plot_height,
        title=f"Sensitivity Analysis: {x_variable} vs {y_variable}",
    )

    weights = [entry["weight"] for entry in data_dict]
    color_mapper = LinearColorMapper(
        palette=Spectral11, low=min(weights), high=max(weights)
    )

    sensitivity_results = []

    for entry in data_dict:
        name = entry["name"]
        weight = entry["weight"]
        data = entry["data"]

        x = data[x_variable]
        y = data[y_variable]

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        total_area = (x_max - x_min) * (y_max - y_min)

        intervals = initialize_intervals(x_min, x_max, num_intervals)
        assign_points_to_intervals(x, y, intervals)
        merged_intervals = merge_intervals(intervals, min_points, x_min, x_max)

        # Calculate sensitivity index only if total_area is not zero
        if total_area != 0:
            sensitivity_index = calculate_sensitivity_index(
                merged_intervals, total_area
            )
        else:
            sensitivity_index = 0

        sensitivity_results.append(
            {
                "name": name,
                "sensitivity": sensitivity_index,
                "merged_intervals": merged_intervals,
                "x_min": x_min,
                "x_max": x_max,
                "x_mean": np.mean(x),
                "y_min": y_min,
                "y_max": y_max,
                "y_mean": np.mean(x),
            }
        )

        create_scatter_plot(p, data, x_variable, y_variable, name, weight, color_mapper)
        color = Spectral11[int(weight * (len(Spectral11) - 1))]
        add_intervals_to_plot(p, merged_intervals, color)

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0, 0))
    p.add_layout(color_bar, "right")

    p.legend.click_policy = "hide"
    p.legend.title = "Datasets"

    if show_graph:
        show(p)

    if export_image:
        export_png(
            p,
            filename=f"{export_path}/{x_variable}_{y_variable}_sensitivity_analysis.png",
        )

    return sensitivity_results


def batch_sensitivity_analysis(
    data_dict: List[Dict[str, Any]],
    x_variables: List[str],
    y_variables: List[str],
    num_intervals: int = 10,
    min_points: int = 5,
    export_image: bool = False,
    export_path: str = ".",
    plot_width: int = 800,
    plot_height: int = 600,
    hover_tool: bool = False,
    show_graph: bool = True,  # Option to show graphs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform batch sensitivity analysis for multiple x and y variable combinations.

    Parameters:
    - data_dict: List of dictionaries containing data and metadata.
    - x_variables: List of x variables to analyze.
    - y_variables: List of y variables to analyze.
    - num_intervals: Number of intervals for sensitivity analysis.
    - min_points: Minimum number of points per interval.
    - export_image: Whether to export the graph as an image.
    - export_path: Path to save the exported image.
    - plot_width: Width of the plot.
    - plot_height: Height of the plot.
    - hover_tool: Whether to add a hover tool to the plot.
    - show_graph: Whether to display the graph.

    Returns:
    - Tuple containing the Pearson correlation matrix and all sensitivity analysis results.
    """
    all_results = {}
    sensitivity_indices = {}  # Store the sensitivity indices for all combinations

    # Run sensitivity analysis for each combination of x and y variables
    for x_var in x_variables:
        for y_var in y_variables:
            print(f"Analyzing sensitivity for {x_var} vs {y_var}...")

            sensitivity_results = sensitivity_analysis(
                data_dict=data_dict,
                x_variable=x_var,
                y_variable=y_var,
                num_intervals=num_intervals,
                min_points=min_points,
                export_image=export_image,
                export_path=export_path,
                plot_width=plot_width,
                plot_height=plot_height,
                hover_tool=hover_tool,
                show_graph=show_graph,  # Pass to inner function
            )

            # Store sensitivity index and map it to the x-y combination
            sensitivity_indices[f"{x_var}_vs_{y_var}"] = [
                result["sensitivity"] for result in sensitivity_results
            ]

            all_results[f"{x_var}_vs_{y_var}"] = sensitivity_results

    # Convert sensitivity indices to DataFrame for correlation calculation
    sensitivity_df = pd.DataFrame(sensitivity_indices)

    # Calculate Pearson correlation matrix
    correlation_matrix = sensitivity_df.corr()

    return correlation_matrix, all_results


def normalize_sensitivity_indices(results):
    """
    Normalize the sensitivity indices for comparison across parameters.
    Normalize each parameter's sensitivity index to the [0, 1] range.
    """
    # Extract all sensitivity indices for normalization
    all_sensitivities = [
        result["sensitivity"]
        for comparison in results.values()
        for result in comparison
    ]

    # Normalize each sensitivity index to the [0, 1] range using Min-Max scaling
    min_sensitivity = min(all_sensitivities)
    max_sensitivity = max(all_sensitivities)

    normalized_results = []
    for comparison, comparison_results in results.items():
        for result in comparison_results:
            normalized_sensitivity = (result["sensitivity"] - min_sensitivity) / (
                max_sensitivity - min_sensitivity
            )
            normalized_results.append(
                {
                    "Comparison": comparison,
                    "Dataset": result["name"],
                    "Sensitivity Index": result["sensitivity"],
                    "Normalized Sensitivity Index": normalized_sensitivity,
                }
            )

    return normalized_results


def print_sensitivity_results(all_results):
    """
    Print sensitivity analysis results in a neat, human-readable format.
    """
    # Prepare a list for storing all the results in a tabular format
    results_list = []

    for comparison, results in all_results.items():
        for result in results:
            results_list.append([comparison, result["name"], result["sensitivity"]])

    # Print the results as a table
    headers = ["Comparison", "Dataset", "Sensitivity Index"]
    print(tabulate(results_list, headers, tablefmt="pretty"))


def filter_results(results, threshold=0.5):
    """
    Filter the sensitivity analysis results to keep only those with a normalized sensitivity index >= threshold.
    Also, return a list of the unique parameters being compared in the results, without prefixes.
    """
    filtered_results = [
        result
        for result in results
        if result["Normalized Sensitivity Index"] >= threshold
    ]

    parameters_set = set()
    for result in filtered_results:
        comparison = result["Comparison"]
        parameters = comparison.split("_vs_")
        parameters_set.update(parameters)

    return filtered_results, list(parameters_set)


def print_normalized_results(normalized_results):
    """
    Print the normalized sensitivity analysis results in a table.
    """
    headers = [
        "Comparison",
        "Dataset",
        "Sensitivity Index",
        "Normalized Sensitivity Index",
    ]
    results_list = []

    for result in normalized_results:
        results_list.append(
            [
                result["Comparison"],
                result["Dataset"],
                f"{result['Sensitivity Index']:.4f}",
                f"{result['Normalized Sensitivity Index']:.4f}",
            ]
        )

    print(tabulate(results_list, headers, tablefmt="pretty"))


def rank_and_print_results(normalized_results):
    """
    Rank the results by sensitivity (descending) while keeping variables (Comparisons) grouped.
    Print the ranked results in a table format.
    """
    # Group results by "Comparison"
    grouped_results = {}
    for result in normalized_results:
        comparison = result["Comparison"]
        if comparison not in grouped_results:
            grouped_results[comparison] = []
        grouped_results[comparison].append(result)

    # Sort each group by "Normalized Sensitivity Index" (descending)
    for comparison, results in grouped_results.items():
        grouped_results[comparison] = sorted(
            results, key=lambda r: r["Normalized Sensitivity Index"], reverse=True
        )

    # Combine the groups back into a single list for printing
    ranked_results = []
    for comparison, results in grouped_results.items():
        ranked_results.extend(results)

    # Print the results
    headers = [
        "Comparison",
        "Dataset",
        "Sensitivity Index",
        "Normalized Sensitivity Index",
    ]
    results_list = [
        [
            result["Comparison"],
            result["Dataset"],
            f"{result['Sensitivity Index']:.4f}",
            f"{result['Normalized Sensitivity Index']:.4f}",
        ]
        for result in ranked_results
    ]
    print(tabulate(results_list, headers, tablefmt="pretty"))


def aggregate_and_rank_results(filtered_results, variables):
    """
    Aggregate sensitivities for each variable comparison and rank them from most sensitive to least sensitive.
    Create tables for each variable dynamically.
    """
    # Aggregate sensitivity by variable comparison
    aggregated_sensitivity = {}
    for result in filtered_results:
        comparison = result["Comparison"]
        sensitivity = result["Sensitivity Index"]
        if comparison not in aggregated_sensitivity:
            aggregated_sensitivity[comparison] = []
        aggregated_sensitivity[comparison].append(sensitivity)

    # Compute average and max sensitivity for ranking
    ranked_comparisons = [
        {
            "Comparison": comparison,
            "Average Sensitivity": sum(sensitivities) / len(sensitivities),
            "Max Sensitivity": max(sensitivities),
        }
        for comparison, sensitivities in aggregated_sensitivity.items()
    ]

    # Sort comparisons by average sensitivity (descending)
    ranked_comparisons.sort(key=lambda x: x["Average Sensitivity"], reverse=True)

    # Print the overall ranking results
    headers = ["Comparison", "Average Sensitivity", "Max Sensitivity"]
    results_list = [
        [
            r["Comparison"],
            f"{r['Average Sensitivity']:.4f}",
            f"{r['Max Sensitivity']:.4f}",
        ]
        for r in ranked_comparisons
    ]
    print("Overall Ranked Comparisons:")
    print(tabulate(results_list, headers, tablefmt="pretty"))

    # Separate and rank results for each variable
    variable_results = {var: [] for var in variables}
    for var in variables:
        variable_results[var] = [
            r for r in ranked_comparisons if r["Comparison"].startswith(f"{var}_vs")
        ]
        variable_results[var].sort(key=lambda x: x["Max Sensitivity"], reverse=True)

    # Print ranked results for each variable
    for var in variables:
        print(f"\nRanked {var.capitalize()} Comparisons:")
        var_headers = ["Comparison", "Average Sensitivity", "Max Sensitivity"]
        var_table = [
            [
                r["Comparison"],
                f"{r['Average Sensitivity']:.4f}",
                f"{r['Max Sensitivity']:.4f}",
            ]
            for r in variable_results[var]
        ]
        print(tabulate(var_table, var_headers, tablefmt="pretty"))

    return ranked_comparisons, variable_results


def plot_y_range_with_trend_lines_and_x_intervals_weighted(
    data_dict, sensitivity_results, x_variable, y_variable
):
    """
    Plot the range of y values for each interval of x as points connected by lines for all datasets,
    dynamically using the number of intervals from sensitivity_analysis results.
    Visualize the x intervals with markers, with color determined by the weight.

    Args:
    - data_dict: List of dictionaries with datasets for analysis.
    - sensitivity_results: Results of sensitivity analysis containing merged intervals and num_intervals.
    - x_variable: Input variable for interval division.
    - y_variable: Output variable whose range is calculated.
    """
    # Initialize the figure
    p = figure(
        width=800,
        height=600,
        title=f"Y Range Across X Intervals with Trend Lines and X Interval Markers",
        x_axis_label=f"{x_variable} Intervals",
        y_axis_label=f"Range of {y_variable}",
    )

    # Loop through each dataset and plot
    for idx, entry in enumerate(data_dict):
        name = entry["name"]
        weight = entry["weight"]

        # Compute the color based on the weight using Spectral11
        color = Spectral11[int(weight * (len(Spectral11) - 1))]

        # Find the sensitivity result for this dataset
        sensitivity_result = next(
            res for res in sensitivity_results if res["name"] == name
        )

        # Extract the merged intervals and number of intervals from the sensitivity result
        merged_intervals = sensitivity_result["merged_intervals"]
        num_intervals = len(
            merged_intervals
        )  # Number of intervals used in this dataset

        # Extract interval centers and y ranges
        interval_centers = [
            (interval["x_min"] + interval["x_max"]) / 2 for interval in merged_intervals
        ]
        y_ranges = [
            interval["y_max"] - interval["y_min"] for interval in merged_intervals
        ]
        x_intervals = [
            interval["x_max"] - interval["x_min"] for interval in merged_intervals
        ]

        # Create ColumnDataSource for this dataset
        source = ColumnDataSource(
            data=dict(
                interval_centers=interval_centers,
                y_ranges=y_ranges,
                weight=[weight] * num_intervals,
            )
        )

        # Add trend lines for the dataset (y-range as a function of x intervals)
        p.line(
            x="interval_centers",
            y="y_ranges",
            source=source,
            line_width=2,
            color=color,  # Use the explicit color based on weight
            legend_label=name,
            line_dash="solid",
        )
        p.circle(
            x="interval_centers",
            y="y_ranges",
            source=source,
            size=6,
            color=color,  # Use the explicit color based on weight
            legend_label=name,
        )

        # # Add visual marker for the x intervals (horizontal bars centered on the points)
        # for i in range(num_intervals):
        #     p.segment(
        #         x0=interval_centers[i] - x_intervals[i] / 2,  # Left side of the x interval
        #         y0=y_ranges[i],  # Y value at the center of the range
        #         x1=interval_centers[i] + x_intervals[i] / 2,  # Right side of the x interval
        #         y1=y_ranges[i],  # Y value at the center of the range
        #         line_width=2,
        #         color=color,  # Use the explicit color based on weight
        #         line_dash="dotted",  # Dotted line for x interval marker
        #         line_cap="round",  # Rounded line cap at the ends of the segment
        #     )

    # Configure legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    # Show the plot
    show(p)


def pinching_sensitivity_analysis_data_dict(
    data_dict,
    x_variable,
    y_variable,
    num_intervals=10,
):
    """
    Perform pinching sensitivity analysis for all datasets in a data_dict.

    Args:
    - data_dict: List of dictionaries, each containing dataset name, weight, and data (as a DataFrame).
    - x_variable: Input variable for sensitivity analysis.
    - y_variable: Output variable to evaluate sensitivity against.
    - num_intervals: Number of intervals for input variable.

    Returns:
    - results_dict: A dictionary with pinching results for each dataset and the overall pinching index.
    """
    overall_results = []

    for entry in data_dict:
        name = entry["name"]
        weight = entry["weight"]
        data = entry["data"]

        x = data[x_variable].to_numpy()
        y = data[y_variable].to_numpy()

        # Define input range and intervals
        x_min, x_max = min(x), max(x)
        interval_width = (x_max - x_min) / num_intervals
        total_output_range = max(y) - min(y)

        # Baseline: Total area across all intervals
        total_area = (x_max - x_min) * total_output_range

        pinching_results = []

        for i in range(num_intervals):
            # Define the interval for x
            x_i_min = x_min + i * interval_width
            x_i_max = x_i_min + interval_width

            # Pinch x to the midpoint of the interval
            x_mid = (x_i_min + x_i_max) / 2

            # Compute the resulting output range when x is fixed to x_mid
            y_pinch = y[(x >= x_i_min) & (x < x_i_max)]
            if len(y_pinch) > 0:
                y_reduced_range = max(y_pinch) - min(y_pinch)
            else:
                y_reduced_range = 0

            # Calculate the reduction in uncertainty
            reduction = total_output_range - y_reduced_range
            pinching_index = (
                reduction / total_output_range if total_output_range > 0 else 0
            )

            pinching_results.append(
                {
                    "Interval": f"[{x_i_min:.2f}, {x_i_max:.2f}]",
                    "Midpoint": x_mid,
                    "Reduced Range": y_reduced_range,
                    "Reduction": reduction,
                    "Pinching Index": pinching_index,
                }
            )

        # Aggregate pinching results for the dataset
        overall_pinching_index = np.mean(
            [r["Pinching Index"] for r in pinching_results]
        )

        overall_results.append(
            {
                "Dataset": name,
                "Weight": weight,
                "Overall Pinching Index": overall_pinching_index,
                "Pinching Results": pinching_results,
            }
        )

        # Print interval-specific results
        print(f"\nPinching Sensitivity Results for Dataset: {name}")
        headers = [
            "Interval",
            "Midpoint",
            "Reduced Range",
            "Reduction",
            "Pinching Index",
        ]
        results_table = [
            [
                r["Interval"],
                f"{r['Midpoint']:.2f}",
                f"{r['Reduced Range']:.2f}",
                f"{r['Reduction']:.2f}",
                f"{r['Pinching Index']:.4f}",
            ]
            for r in pinching_results
        ]
        print(tabulate(results_table, headers, tablefmt="pretty"))

    # Return aggregated results for all datasets
    return overall_results


def normalize_to_global_min_max(data_dict, selected_params):
    """
    Normalize the design data based on the global min and max for each parameter.
    The scale is normalized such that 0 corresponds to the global min,
    and 1 corresponds to the global max for each parameter.

    Parameters:
    - data_dict: List of dictionaries containing 'name', 'weight', and 'data' (DataFrame).
    - selected_params: List of parameter names to include in the plot.

    Returns:
    - normalized_data: Dictionary with the normalized design data.
    - global_min_values: Global minimum values for each parameter across datasets.
    - global_max_values: Global maximum values for each parameter across datasets.
    """
    global_min_values = {}
    global_max_values = {}

    # Compute global min and max values for each parameter
    for param in selected_params:
        all_values = pd.concat([dataset["data"][param] for dataset in data_dict])
        global_min_values[param] = all_values.min()
        global_max_values[param] = all_values.max()

    normalized_data = {}
    for dataset in data_dict:
        df = dataset["data"]
        normalized_data[dataset["name"]] = {}
        for param in selected_params:
            param_values = df[param]
            if global_min_values[param] == global_max_values[param]:
                normalized_data[dataset["name"]][param] = np.full_like(
                    param_values, 1.0
                )
            else:
                normalized_data[dataset["name"]][param] = (
                    param_values - global_min_values[param]
                ) / (global_max_values[param] - global_min_values[param])

    return normalized_data, global_min_values, global_max_values


def create_comparative_radar_plot(
    data_dict, selected_params, figsize=(26, 26), title="Radar Plot of Normalized Means"
):
    """
    Create a radar plot comparing multiple design sets (DataFrames) showing normalized mean values.

    Parameters:
    - data_dict: List of dictionaries containing 'name', 'weight', and 'data' (DataFrame).
    - selected_params: List of parameter names to include in the plot.
    - title: Title of the radar plot.
    """
    # Normalize the design data using global min-max values
    normalized_data, global_min_values, global_max_values = normalize_to_global_min_max(
        data_dict, selected_params
    )

    # Prepare angles for the radar plot
    num_vars = len(selected_params)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar plot

    # Create radar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Iterate through each dataset and plot the normalized mean values
    for dataset in data_dict:
        name = dataset["name"]
        weight = dataset["weight"]

        # Get the normalized data for the current dataset
        normalized_means = [
            normalized_data[name][param].mean() for param in selected_params
        ]

        # Append the first value to close the loop
        normalized_means = np.append(normalized_means, normalized_means[0])

        # Plot the radar chart for the current dataset
        ax.plot(
            angles,
            normalized_means,
            label=name,
            marker="o",
            linestyle="-",
            color=plt.cm.viridis(weight),
        )
        ax.fill(angles, normalized_means, alpha=0.25, color=plt.cm.viridis(weight))

    # Update labels to include the global min and max values for each parameter in scientific notation
    xticks_labels = []
    for param in selected_params:
        # Get min and max values for this parameter
        min_val = global_min_values[param] if param in global_min_values else "N/A"
        max_val = global_max_values[param] if param in global_max_values else "N/A"
        # Get description and trim it
        variable_description = get_description(param)
        variable_description = trim_description(variable_description)
        variable_description = trim_to_nearest_word(variable_description, max_length=40)

        # Format the labels
        if min_val != "N/A" and max_val != "N/A":
            xticks_labels.append(
                f"{variable_description}\nMin: {min_val:.1e}\nMax: {max_val:.1e}"
            )
        else:
            xticks_labels.append(f"{variable_description}\nData not available")

    # Set the labels for each angle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticks_labels, fontsize="large")

    # Title and legend
    ax.set_title(title, fontsize=15, pad=20)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize="large",
        title="Design Sets",
        title_fontsize="large",
    )

    # Show the plot
    plt.tight_layout()
    plt.show()


variable_descriptions_path = (
    "/home/graeme/uq-sa-fusion-optimisation/uq_tools/variable_descriptions.json"
)
# Open and read the JSON file
with open(variable_descriptions_path, "r") as file:
    variable_descriptions = json.load(file)


def get_description(variable_name):
    for variables in variable_descriptions.values():
        if variable_name in variables:
            return variables[variable_name]
    return None  # or a default message if description not found


def trim_description(description):
    # Remove any text inside parentheses or brackets, including the parentheses or brackets themselves
    trimmed = re.sub(r"\s*\(.*?\)|\s*\[.*?\]", "", description)
    return trimmed.strip()


def trim_to_nearest_word(description, max_length=30):
    # If the description is already within the limit, return it as-is
    if len(description) <= max_length:
        return description
    # Find the maximum index where the substring has no cut-off words
    trimmed_description = description[:max_length].rsplit(" ", 1)[0]
    return trimmed_description + "..."
