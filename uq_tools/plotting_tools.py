import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import export_png, export_svg
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.palettes import Spectral11
from statsmodels.api import distributions as sm_distributions
from math import pi
import seaborn as sns
from uncertainty_data import process_variable_dict
from matplotlib.colors import to_rgb, to_hex
from tabulate import tabulate  # If you want a neat table format


def plot_sobol_indices(sensitivity_df, figure_of_merit, significance_level=0.05):
    """Plot the Sobol Indices from a sensitivity analysis."""
    fig, ax = plt.subplots(1)
    ax.tick_params(labelsize=14)

    sensitivity_df.plot(
        kind="barh", y="S1", xerr="S1_conf", ax=ax, align="center", capsize=3
    )

    ax.set_xlabel(f"Sensitivity Indices: {figure_of_merit}", fontsize=20)
    ax.set_ylabel("PROCESS parameter", fontsize=20)

    ax.fill_betweenx(
        y=[-0.5, len(sensitivity_df)],
        x1=0,
        x2=significance_level,
        color="grey",
        alpha=0.2,
        hatch="//",
        edgecolor="white",
    )

    plt.grid()
    plt.show()


def plot_sensitivity(
    indices,
    names,
    significance_level=0.05,
    title=None,
    export_image=False,
    export_path=".",
):
    """Plot the results of a sensitivity analysis."""
    p = figure(
        title=title or "Sensitivity Analysis", width=800, height=400, y_range=names
    )

    p.quad(
        top=len(indices),
        bottom=-0.5,
        left=0,
        right=significance_level,
        fill_color="grey",
        fill_alpha=0.2,
    )

    p.hbar(y=names, left=0, right=indices, height=0.8, fill_color="blue", alpha=0.5)

    p.xaxis.axis_label = "Index"
    p.yaxis.axis_label = "PROCESS Parameter"
    p.legend.location = "top_right"
    p.legend.label_text_font_size = "12pt"

    show(p)

    if export_image:
        filename = f"{export_path}/{title or 'sensitivity_plot'}.svg"
        export_svg(p, filename=filename)


def scatter_plot(
    data,
    x_variable,
    y_variable,
    scatter=True,
    hist=True,
    bins=10,
    export_image=False,
    hover_tool=False,
    export_path=".",  # New argument for export path
):
    """Create a scatter plot for two variables.
    Also calculates a 2D histogram to plot the density of points on the x-y axis.

    :param data: Dataframe containing variable data
    :type data: pd.DataFrame
    :param x_variable: x-axis variable
    :type x_variable: str
    :param y_variable: y-axis variable
    :type y_variable: str
    :param scatter: Plot scattered points, defaults to True
    :type scatter: bool, optional
    :param hist: Plot 2D histogram (density of points), defaults to True
    :type hist: bool, optional
    :param bins: Number of bins for 2D histogram, defaults to 10
    :type bins: int, optional
    :param export_image: Save the image, defaults to False
    :type export_image: bool, optional
    :param hover_tool: Include hover tool, defaults to False
    :type hover_tool: bool, optional
    :param export_path: Path for exporting images, defaults to current directory
    :type export_path: str, optional
    """
    # Create a Bokeh figure
    p = figure(title=f"Scatter Plot: {x_variable} vs {y_variable}")

    # Extract data
    x = data[x_variable]
    y = data[y_variable]

    # Compute 2D histogram
    H, xe, ye = np.histogram2d(x=x, y=y, bins=bins)

    # Create an image plot
    if hist:
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
    if scatter:
        scatter_source = ColumnDataSource(data=dict(x=x, y=y))
        p.scatter(x="x", y="y", source=scatter_source, size=8, color="blue", alpha=0.5)

        # Optionally add HoverTool
        if hover_tool:
            hover = HoverTool()
            hover.tooltips = [
                ("Index", "$index"),
                (f"({x_variable}, {y_variable})", "(@x, @y)"),
            ]
            p.add_tools(hover)

    # Customize the plot
    p.xaxis.axis_label = x_variable
    p.yaxis.axis_label = y_variable

    # Show the plot
    show(p)

    if export_image:
        filename = f"{export_path}/{x_variable}_{y_variable}_plot.png"
        export_png(p, filename=filename)


def scatter_grid(
    data,
    variables,
    bins=10,
    scatter=True,
    hist=True,
    height_width=250,
    export_image=False,
    export_path=".",
    hover_tool=False,
):
    """Create a scatter grid for multiple variables."""
    plots = []
    for i, var1 in enumerate(variables):
        row_plots = []
        for j, var2 in enumerate(variables):
            if j >= i:
                H, xe, ye = np.histogram2d(x=data[var1], y=data[var2], bins=bins)
                p = figure(
                    title=f"{var1} vs {var2}", height=height_width, width=height_width
                )

                if hist:
                    p.image(
                        image=[H.T],
                        x=xe[0],
                        y=ye[0],
                        dw=xe[-1] - xe[0],
                        dh=ye[-1] - ye[0],
                        palette="Spectral11",
                        alpha=0.6,
                    )

                if scatter:
                    scatter_source = ColumnDataSource(
                        data={"x": data[var1], "y": data[var2]}
                    )
                    p.scatter(
                        x="x",
                        y="y",
                        source=scatter_source,
                        size=8,
                        color="blue",
                        alpha=0.5,
                    )
                    if hover_tool:
                        hover = HoverTool()
                        hover.tooltips = [
                            ("Index", "$index"),
                            (var1, "@x"),
                            (var2, "@y"),
                        ]
                        p.add_tools(hover)

                p.xaxis.axis_label = var1
                p.yaxis.axis_label = var2
                row_plots.append(p)
            else:
                row_plots.append(None)

        plots.append(row_plots)

    grid = gridplot(plots)
    show(grid)

    if export_image:
        export_png(grid, filename=f"{export_path}/scatter_grid.png")


def weighted_multi_scatter(
    data_dict,
    x_variable,
    y_variable,
    export_image=False,
    export_path=".",
    plot_width=800,
    plot_height=600,
    hover_tool=False,
):
    """Create a scatter plot for two variables from multiple weighted dataframes."""
    p = figure(
        width=plot_width, height=plot_height, title=f"{x_variable} vs {y_variable}"
    )

    weights = [entry["weight"] for entry in data_dict]
    color_mapper = LinearColorMapper(
        palette=Spectral11, low=min(weights), high=max(weights)
    )

    for entry in data_dict:
        name = entry["name"]
        weight = entry["weight"]
        data = entry["data"]

        scatter_source = ColumnDataSource(
            data=dict(
                x=data[x_variable], y=data[y_variable], weight=[weight] * len(data)
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

        if hover_tool:
            hover = HoverTool(
                tooltips=[
                    ("Index", "$index"),
                    (x_variable, "@x"),
                    (y_variable, "@y"),
                    ("Weight", "@weight"),
                ]
            )
            p.add_tools(hover)

    p.add_layout(ColorBar(color_mapper=color_mapper), "right")
    p.legend.click_policy = "hide"
    show(p)

    if export_image:
        export_png(
            p, filename=f"{export_path}/{x_variable}_{y_variable}_weighted_plot.png"
        )


def radar_plot(design_data, parameters):
    # Number of parameters
    num_vars = len(parameters)

    # Compute angles for the parameters
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Set the color palette
    colors = sns.color_palette("husl", len(design_data))

    # Plot each design set
    for i, (set_name, data) in enumerate(design_data.items()):
        # Get mean and lower/upper bounds
        mean_values = data["mean"] + [data["mean"][0]]  # Close the loop
        lower_values = data["lower"] + [data["lower"][0]]
        upper_values = data["upper"] + [data["upper"][0]]

        # Plot the mean
        ax.plot(angles, mean_values, label=set_name, color=colors[i])

        # Plot the lower and upper bounds as error bars
        # ax.fill(angles, upper_values, color=colors[i], alpha=0.2)
        # ax.fill(angles, lower_values, color=colors[i], alpha=0.2)

    # Add labels for each parameter
    plt.xticks(angles[:-1], parameters)

    # Add title and legend
    plt.title("Radar Plot Comparison of Design Sets", size=15, color="black")
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    # Show the plot
    plt.show()


def normalize_parameters_relative_to_max(data_dicts):
    # Find the maximum design_range_end across all datasets
    normalized_data = {}

    # Normalize design_mean for each parameter relative to its design_range_end
    for ds_name, design_set in data_dicts.items():
        for var, values in design_set.items():
            if "design_mean" in values and "design_range_end" in values:
                current_mean = values["design_mean"]
                design_range_end = values["design_range_end"]
                # Normalize using the corresponding design_range_end
                if design_range_end != 0:
                    normalized_value = current_mean / design_range_end
                else:
                    normalized_value = 0  # If the design_range_end is zero

                normalized_data.setdefault(ds_name, {})[var] = normalized_value

    return normalized_data


# Normalize the data
def normalize_design_data(design_data):
    # Create a structure to hold max values for each variable

    min_values = {}
    max_values = {}

    # Gather min and max values for each parameter
    for set_name, set_data in design_data.items():
        for var_name, values in set_data.items():
            # Initialize or update the max values
            if var_name not in max_values:
                max_values[var_name] = {
                    "mean": values["design_mean"],
                    "lower": values["design_range_start"],
                    "upper": values["design_range_end"],
                }
            else:
                max_values[var_name]["mean"] = max(
                    max_values[var_name]["mean"], values["design_mean"]
                )
                max_values[var_name]["lower"] = max(
                    max_values[var_name]["lower"], values["design_range_start"]
                )
                max_values[var_name]["upper"] = max(
                    max_values[var_name]["upper"], values["design_range_end"]
                )

            # Initialize or update the min values
            if var_name not in min_values:
                min_values[var_name] = {
                    "lower": values["design_range_start"],
                    "mean": values["design_mean"],
                    "upper": values["design_range_end"],
                }
            else:
                min_values[var_name]["lower"] = min(
                    min_values[var_name]["lower"], values["design_range_start"]
                )
                min_values[var_name]["mean"] = min(
                    min_values[var_name]["mean"], values["design_mean"]
                )
                min_values[var_name]["upper"] = min(
                    min_values[var_name]["upper"], values["design_range_end"]
                )

    # Normalize the design data based on each variable's max values
    normalized_data = {}
    for set_name, set_data in design_data.items():
        normalized_data[set_name] = {}
        for var_name, values in set_data.items():
            design_mean = values["design_mean"]
            normalized_data[set_name][var_name] = {
                "mean": (
                    (design_mean - min_values[var_name]["lower"])
                    / (max_values[var_name]["upper"] - min_values[var_name]["lower"])
                    if max_values[var_name]["mean"] != 0
                    else 0
                ),
                "lower": (
                    max_values[var_name]["lower"] / max_values[var_name]["lower"]
                    if max_values[var_name]["lower"] != 0
                    else 0
                ),
                "upper": (
                    max_values[var_name]["lower"] / max_values[var_name]["lower"]
                    if max_values[var_name]["lower"] != 0
                    else 0
                ),
                "robustness": values["robustness"],  # Leave robustness as is for now
            }

    return normalized_data


def plot_radar_charts(data, normalized_data, selected_parameters=None):
    if selected_parameters is None:
        selected_parameters = list(next(iter(normalized_data.values())).keys())

    num_vars = len(selected_parameters)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot each design set
    for design_set_name, values in normalized_data.items():
        # Get selected mean values for plotting
        selected_values = [
            values[param]["mean"] for param in selected_parameters if param in values
        ]
        selected_values += selected_values[:1]  # Close the circle

        # Plot the radar chart
        ax.fill(angles, selected_values, alpha=0.25, label=design_set_name)
        ax.plot(angles, selected_values, linewidth=2)  # Outline

    # Update the labels to include the max design_range_end values in scientific notation
    updated_labels = []
    for param in selected_parameters:
        # Get max design_range_end value for this parameter across design sets
        design_range_end_values = [
            data[design_set][param]["design_range_end"]
            for design_set in data
            if param in data[design_set]
        ]
        max_value = max(design_range_end_values) if design_range_end_values else 0
        # Append max value to label in scientific notation
        updated_labels.append(f"{param} (max: {max_value:.1e})")
    process_variable_dict[param]
    # Set the updated labels for each angle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(updated_labels)

    # Title and legend
    plt.title("Radar Plot Comparison of Design Sets")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.show()


def darken_color_by_weight(color, factor=0.7):
    """Darken a given hex color by a specified factor."""
    rgb = to_rgb(color)  # Convert to RGB tuple
    darker_rgb = tuple([c * factor for c in rgb])  # Scale RGB values
    return to_hex(darker_rgb)  # Convert back to hex


def sensitivity_analysis(
    data_dict,
    x_variable,
    y_variable,
    num_intervals=10,
    min_points=5,  # Minimum number of points per interval
    export_image=False,
    export_path=".",
    plot_width=800,
    plot_height=600,
    hover_tool=False,
    show_graph=True,  # Option to show the graph
):
    """Perform sensitivity analysis, calculate sensitivity indices, and merge intervals with insufficient points."""
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

        x = data[x_variable].to_numpy()
        y = data[y_variable].to_numpy()

        # Sensitivity Analysis Logic
        x_min, x_max = min(x), max(x)
        interval_width = (x_max - x_min) / num_intervals
        total_area = (x_max - x_min) * (max(y) - min(y))
        scatter_area = 0

        # Initialize intervals
        intervals = [
            {
                "x_min": x_min + i * interval_width,
                "x_max": x_min + (i + 1) * interval_width,
                "points": [],
            }
            for i in range(num_intervals)
        ]

        # Assign points to intervals
        for xi, yi in zip(x, y):
            for interval in intervals:
                if interval["x_min"] <= xi < interval["x_max"]:
                    interval["points"].append(yi)
                    break

        # Merge intervals with fewer than `min_points`
        merged_intervals = []
        temp_points = []
        for interval in intervals:
            temp_points.extend(interval["points"])
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
                    }
                )
                temp_points = []  # Reset temp_points after merging
        # Handle leftover points
        if temp_points:
            if merged_intervals:
                merged_intervals[-1]["y_max"] = max(
                    merged_intervals[-1]["y_max"], max(temp_points)
                )
            else:
                y_min, y_max = min(temp_points), max(temp_points)
                merged_intervals.append(
                    {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
                )

        # Calculate scatter area
        for interval in merged_intervals:
            interval_width = interval["x_max"] - interval["x_min"]
            interval_height = interval["y_max"] - interval["y_min"]
            scatter_area += interval_width * interval_height

        sensitivity_index = 1 - (scatter_area / total_area)
        sensitivity_results.append(
            {
                "name": name,
                "sensitivity": sensitivity_index,
                "merged_intervals": merged_intervals,  # Add merged intervals to results
            }
        )
        # Visualization Logic
        scatter_source = ColumnDataSource(
            data=dict(
                x=data[x_variable], y=data[y_variable], weight=[weight] * len(data)
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

        # Map weight to color explicitly
        color = Spectral11[int(weight * (len(Spectral11) - 1))]

        for interval in merged_intervals:
            # Add a semi-transparent shaded box
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
                fill_alpha=0.2,  # Low alpha for transparency
                fill_color=color,  # Explicit color mapping
            )

            # Add an outline for the box
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
                line_color="black",  # Outline color
                line_width=1,  # Outline width
                fill_alpha=0,  # Transparent fill for outline
            )

    # Add color bar
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0, 0))
    p.add_layout(color_bar, "right")

    # Configure legend
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
    data_dict,
    x_variables,
    y_variables,
    num_intervals=10,
    min_points=5,
    export_image=False,
    export_path=".",
    plot_width=800,
    plot_height=600,
    hover_tool=False,
    show_graph=True,  # Option to show graphs
):
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

    # Alternatively, print it in a simple format if tabulate is not available
    # for comparison, results in all_results.items():
    #     print(f"Results for {comparison}:")
    #     for result in results:
    #         print(f"  {result['name']} - Sensitivity Index: {result['sensitivity']:.4f}")
    #     print("\n")


def filter_results(results, threshold=0.5):
    """
    Filter the sensitivity analysis results to keep only those with a normalized sensitivity index >= threshold.
    """
    return [
        result
        for result in results
        if result["Normalized Sensitivity Index"] >= threshold
    ]


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


def aggregate_and_rank_results(filtered_results):
    """
    Aggregate sensitivities for each variable comparison (e.g., aspect_vs_<variable>)
    and rank them from most sensitive to least sensitive.
    Also, rank variables for aspect and tbrnmn separately.
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

    # Separate and rank aspect and tbrnmn results
    aspect_results = [
        r for r in ranked_comparisons if r["Comparison"].startswith("aspect_vs")
    ]
    burn_time_results = [
        r for r in ranked_comparisons if r["Comparison"].startswith("tbrnmn_vs")
    ]

    # Sort within each group by Max Sensitivity
    aspect_results.sort(key=lambda x: x["Max Sensitivity"], reverse=True)
    burn_time_results.sort(key=lambda x: x["Max Sensitivity"], reverse=True)

    # Print ranked results for aspect and burn time
    print("\nRanked Aspect Comparisons:")
    aspect_headers = ["Comparison", "Average Sensitivity", "Max Sensitivity"]
    aspect_table = [
        [
            r["Comparison"],
            f"{r['Average Sensitivity']:.4f}",
            f"{r['Max Sensitivity']:.4f}",
        ]
        for r in aspect_results
    ]
    print(tabulate(aspect_table, aspect_headers, tablefmt="pretty"))

    print("\nRanked Burn Time Comparisons:")
    burn_time_headers = ["Comparison", "Average Sensitivity", "Max Sensitivity"]
    burn_time_table = [
        [
            r["Comparison"],
            f"{r['Average Sensitivity']:.4f}",
            f"{r['Max Sensitivity']:.4f}",
        ]
        for r in burn_time_results
    ]
    print(tabulate(burn_time_table, burn_time_headers, tablefmt="pretty"))
    print(tabulate(burn_time_table, burn_time_headers, tablefmt="pretty"))

    return ranked_comparisons, aspect_results, burn_time_results


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
