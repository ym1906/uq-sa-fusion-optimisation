import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import export_png, export_svg
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.palettes import Spectral11
from statsmodels.api import distributions as sm_distributions


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
