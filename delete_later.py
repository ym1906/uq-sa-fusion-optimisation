    def plot_network(
        self,
        correlation_matrix=None,
        threshold=0.1,
        variables=None,
        nodes_to_highlight=None,
        figsize=(12, 12),
    ):
        if correlation_matrix is None:
            correlation_matrix = self.correlation_matrix()

        # Create a graph from the correlation matrix
        G = nx.Graph()
        p = figure()
        # Add nodes
        G.add_nodes_from(correlation_matrix.columns)

        # Add edges with weights (correlations)
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # To avoid duplicate edges
                    correlation_value = correlation_matrix.iloc[i, j]
                    if abs(correlation_value) >= threshold:
                        G.add_edge(col1, col2, weight=correlation_value)

        # Define layout (spring_layout is just an example, you can choose other layouts)
        layout = nx.spring_layout(G, k=0.7)

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

        # Separate positive and negative edges in the subgraph
        positive_edges = [
            (u, v) for u, v, d in subgraph.edges(data=True) if d["weight"] > 0
        ]
        negative_edges = [
            (u, v) for u, v, d in subgraph.edges(data=True) if d["weight"] < 0
        ]

        # Choose a seaborn color palette
        color_palette = sns.color_palette("husl", len(subgraph.nodes))
        # Create custom legend lines
        legend_lines = [
            Line2D(
                [0],
                [0],
                color=color_palette[0],
                linewidth=2,
                label="Positive Correlation",
            ),
            Line2D(
                [0], [0], color="tab:blue", linewidth=2, label="Negative Correlation"
            ),
        ]
        # Draw positive edges in red
        nx.draw_networkx_edges(
            subgraph,
            pos=layout,
            edgelist=positive_edges,
            edge_color=[
                (1, 0, 0, abs(subgraph[u][v]["weight"])) for u, v in positive_edges
            ],
            width=8.0,
        )
        nx.draw_networkx_edges(
            subgraph,
            pos=layout,
            edgelist=positive_edges,
            edge_color="tab:grey",
            width=1.0,
            alpha=1.0,
        )

        # Draw negative edges in blue
        nx.draw_networkx_edges(
            subgraph,
            pos=layout,
            edgelist=negative_edges,
            edge_color=[
                (0, 0, 1, abs(subgraph[u][v]["weight"])) for u, v in negative_edges
            ],
            width=8.0,
        )
        nx.draw_networkx_edges(
            subgraph,
            pos=layout,
            edgelist=negative_edges,
            edge_color="tab:grey",
            width=1.0,
            alpha=0.5,
        )
        node_sizes = [subgraph.degree(node) * 800 for node in subgraph.nodes]

        # Draw nodes with labels
        if nodes_to_highlight is not None:
            node_colors = [
                "lightgray" if node not in nodes_to_highlight else "tab:green"
                for node in subgraph.nodes
            ]
        else:
            node_colors = "lightgray"
        options = {"edgecolors": "tab:gray"}
        nx.draw_networkx_nodes(
            subgraph,
            pos=layout,
            node_size=node_sizes,
            node_color=node_colors,
            **options,
        )
        nx.draw_networkx_labels(
            subgraph,
            pos=layout,
            font_size=10,
            font_color="black",
            font_weight="bold",
        )

        # Draw edge labels separately
        edge_labels = {
            (i, j): f"{subgraph[i][j]['weight']:.2f}" for i, j in subgraph.edges()
        }
        nx.draw_networkx_edge_labels(
            subgraph,
            pos=layout,
            edge_labels=edge_labels,
            font_size=6,
            font_color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="none",
                facecolor=color_palette[0]
                if edge_labels.get((i, j), 0) > 0
                else color_palette[1],
                alpha=0.1,  # Background alpha
            ),
        )

        # Add legend
        plt.legend(handles=legend_lines)

        # Add title with included variables and nodes
        plt.title(
            f"Correlation Network Plot\nIncluded Variables: {title_variables}\nHighlighted Nodes: {nodes_to_highlight}"
        )

        # Set the figure size
        plt.gcf().set_size_inches(figsize[0], figsize[1])

        plt.show()

        p = figure()
        p.grid.grid_line_color = Nonecolor n in enumerate(G.nodes))
        H = nx.relabel_nodes(G, mapping)
        graph = from_networkx(H, nx.spring_layout, scale=1.8, center=(0, 0))
        p.renderers.append(graph)
        graph.node_renderer.glyph = Circle(size=circle_size)

        show(p)