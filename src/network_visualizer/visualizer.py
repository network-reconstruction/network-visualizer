"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = network_visualizer.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""
import logging
from network_visualizer import __version__

__author__ = "yao-creative"
__copyright__ = "yao-creative"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from network_visualizer.skeleton import fib`,
# when using this Python module as a library.


import networkx as nx
import numpy as np
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
from typing import Optional, Union

class NetworkVisualizer:
    """
    A class to visualize network graphs using NetworkX, PyVis, and Seaborn.
    It supports visualization from either a NetworkX graph object, an adjacency matrix (with or without labels), or a CSV file.

    Attributes:
        G (networkx.Graph): The graph to be visualized.
        directed (bool): Indicates if the graph is directed.
        weighted (bool): Indicates if the graph is weighted.
        verbose (bool): Enables verbose output for debugging and informational purposes.
    """

    def __init__(self, 
                 graph: Optional[nx.Graph] = None, 
                 adjacency_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                 adjacency_list: Optional[Union[str, list]] = None,
                 csv_path: Optional[str] = None, 
                 directed: bool = False, 
                 weighted: bool = False, 
                 verbose: bool = False,
                 create_graph: bool = False) -> None:
        """
        Initializes the NetworkVisualizer with a graph, an adjacency matrix, an adjacency list, or a CSV file.

        Args:
            graph (networkx.Graph, optional): A NetworkX graph object.
            adjacency_matrix (numpy.ndarray or pandas.DataFrame, optional): An adjacency matrix.
            adjacency_list (str or list, optional): An adjacency list or a path to an adjacency list file.
            csv_path (str, optional): Path to a CSV file containing edge list or adjacency matrix with labels.
            directed (bool): Whether the graph is directed. Default is False.
            weighted (bool): Whether the graph is weighted. Default is False.
            verbose (bool): Whether to enable verbose output. Default is False.
            create_graph (bool): Whether to create a graph from the adjacency matrix, adjacency list, or CSV file. Default is False.

        Raises:
            ValueError: If neither graph, adjacency matrix, adjacency list, nor CSV path is provided.
        """
        self.verbose = verbose

        if graph:
            self.G = graph
            if self.verbose:
                print("Graph provided directly.")
            self.adj_matrix = jnp.array(nx.to_numpy_array(self.G))
        elif adjacency_matrix is not None:
            if create_graph:
                self.G = self._create_graph_from_adj_matrix(adjacency_matrix, directed, weighted)
                if self.verbose:
                    print("Graph created from adjacency matrix.")
            else:
                self.G = None
            self.adj_matrix = jnp.array(adjacency_matrix)
            if self.verbose:
                print("Adjacency matrix provided.")
        elif adjacency_list is not None:
            if create_graph:
                self.G = self._create_graph_from_adj_list(adjacency_list, directed, weighted)
                if self.verbose:
                    print("Graph created from adjacency list.")
            else:
                self.G = None
            if isinstance(adjacency_list, str):
                with open(adjacency_list, 'r') as f:
                    self.adj_list = f.readlines()
            else:
                self.adj_list = adjacency_list
            if self.verbose:
                print("Adjacency list provided.")
        elif csv_path is not None:
            if create_graph:
                self.G = self._load_graph_from_csv(csv_path, directed, weighted)
                if self.verbose:
                    print(f"Graph loaded from CSV file: {csv_path}")
            else:
                self.G = None
            self.adj_matrix = jnp.array(pd.read_csv(csv_path))
        else:
            raise ValueError("Either graph, adjacency matrix, adjacency list, or CSV path must be provided.")
        
        self.directed = directed
        self.weighted = weighted

    def _create_graph_from_adj_matrix(self, adjacency_matrix: Union[np.ndarray, pd.DataFrame], directed: bool, weighted: bool) -> nx.Graph:
        """
        Creates a graph from an adjacency matrix.

        Args:
            adjacency_matrix (numpy.ndarray or pandas.DataFrame): The adjacency matrix.
            directed (bool): Whether the graph is directed.
            weighted (bool): Whether the graph is weighted.

        Returns:
            networkx.Graph: The graph created from the adjacency matrix.
        """
        if isinstance(adjacency_matrix, pd.DataFrame):
            G = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph() if directed else nx.Graph())
        else:
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph() if directed else nx.Graph())
        return G

    def _create_graph_from_adj_list(self, adjacency_list: Union[str, list], directed: bool, weighted: bool) -> nx.Graph:
        """
        Creates a graph from an adjacency list.

        Args:
            adjacency_list (str or list): The adjacency list or a path to an adjacency list file.
            directed (bool): Whether the graph is directed.
            weighted (bool): Whether the graph is weighted.

        Returns:
            networkx.Graph: The graph created from the adjacency list.
        """
        if isinstance(adjacency_list, str):
            G = nx.read_adjlist(adjacency_list, create_using=nx.DiGraph() if directed else nx.Graph(), nodetype=int)
        else:
            G = nx.parse_adjlist(adjacency_list, create_using=nx.DiGraph() if directed else nx.Graph(), nodetype=int)
        return G

    def _load_graph_from_csv(self, csv_path: str, directed: bool, weighted: bool) -> nx.Graph:
        """
        Loads a graph from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            directed (bool): Whether the graph is directed.
            weighted (bool): Whether the graph is weighted.

        Returns:
            networkx.Graph: The graph created from the CSV file.
        """
        df = pd.read_csv(csv_path)
        if 'source' in df.columns and 'target' in df.columns:
            if self.verbose:
                print("CSV file contains edge list.")
                print(df.head())
            if weighted:
                G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph() if directed else nx.Graph())
            else:
                G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph() if directed else nx.Graph())
        else:
            if self.verbose:
                print("CSV file contains adjacency matrix with labels.")
            G = self._create_graph_from_adj_matrix(df, directed, weighted)
        return G

    def get_info(self) -> str:
        """
        Returns basic information about the graph.

        Returns:
            str: Information about the graph.
        """
        info = nx.info(self.G)
        if self.verbose:
            print(info)
        return info

    def save_network(self, output_dir: str, **kwargs) -> None:
        """
        Saves an interactive visualization of the network as an HTML file in the specified output directory.

        Args:
            output_dir (str): The directory to save the network visualization.
            **kwargs: Additional keyword arguments to customize the network visualization. 
                    Possible arguments include:
                    - layout (str): Layout of the network (e.g., 'hierarchical', 'random', etc.).
                    - notebook (bool): Whether to display in a Jupyter notebook (default: True).
                    - directed (bool): Whether the network is directed (default: self.directed).
                    - weighted (bool): Whether the network is weighted (default: self.weighted).
                    - embedding (str): Embedding style for nodes (e.g., 'spring', 'shell', etc.).

        """
        if self.verbose:
            print(f"Saving network visualization to directory: {output_dir}")

        # Extract parameters from kwargs with default fallbacks
        layout = kwargs.get('layout', None)
        notebook = kwargs.get('notebook', True)
        directed = kwargs.get('directed', self.directed)
        weighted = kwargs.get('weighted', self.weighted)
        embedding = kwargs.get('embedding', None)

        net = Network(notebook=notebook, directed=directed)

        if self.G is None:
            self.G = self._create_graph_from_adj_matrix(self.adj_matrix, directed, weighted)

        for node in self.G.nodes:
            net.add_node(node, label=str(node))

        for edge in self.G.edges:
            if weighted:
                net.add_edge(edge[0], edge[1], weight=self.G[edge[0]][edge[1]]['weight'])
            else:
                net.add_edge(edge[0], edge[1])

        # Apply layout if specified
        if layout:
            net.repulsion()
            if layout == 'hierarchical':
                net.hierarchical_repulsion()
            elif layout == 'random':
                net.force_atlas_2based()
            # Add other layout options as needed

        # Apply embedding if specified
        if embedding:
            if embedding == 'spring':
                pos = nx.spring_layout(self.G)
            elif embedding == 'shell':
                pos = nx.shell_layout(self.G)
            # Add other embedding styles as needed
            for node, position in pos.items():
                net.nodes[node]['x'] = position[0] * 1000
                net.nodes[node]['y'] = position[1] * 1000

        net.show(f'{output_dir}/network.html')

        if self.verbose:
            print("Network visualization saved.")

    def save_heatmap(self, output_dir: str, title: str = 'Adjacency Matrix Heatmap', max_labels: int = 10) -> None:
        """
        Saves a heatmap of the adjacency matrix as an image file in the specified output directory.

        Args:
            output_dir (str): The directory to save the heatmap image.
            title (str): The title of the heatmap.
            max_labels (int): The maximum number of labels to display per axis.
        """
        if self.verbose:
            print(f"Saving heatmap to directory: {output_dir}")

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(self.adj_matrix, annot=False, cbar=True, xticklabels=True, yticklabels=True, center=0)
        plt.title(title)
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')

        # Set sparse labels
        if len(self.adj_matrix) > max_labels:
            step = max(1, len(self.adj_matrix) // max_labels)
            ax.set_xticks(np.arange(0, len(self.adj_matrix), step))
            ax.set_xticklabels(np.arange(0, len(self.adj_matrix), step))
            ax.set_yticks(np.arange(0, len(self.adj_matrix), step))
            ax.set_yticklabels(np.arange(0, len(self.adj_matrix), step))

        plt.gca().invert_yaxis()  # Invert the y-axis to make the bottom-left corner (0,0)
        plt.savefig(f'{output_dir}/heatmap.png')
        plt.close()
        
        if self.verbose:
            print("Heatmap saved.")

    def _ccdf(self, data):
        """
        Computes the complementary cumulative distribution function (CCDF) for a dataset.

        Args:
            data (array-like): The input data.

        Returns:
            x (array-like): The sorted data points.
            y (array-like): The CCDF values.
        """
        sorted_data = np.sort(data)
        ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, ccdf

    def save_distributions(self, output_dir: str, title: str = '', loglog: bool = True) -> None:
        """
        Saves the degree distributions of the graph based on its type (directed, weighted, etc.)
        as image files in the specified output directory.

        Args:
            output_dir (str): The directory to save the distribution plots.
            title (str): The prefix for the titles of the plots.
            loglog (bool): Whether to plot scatter plots on a log-log scale. Default is True.
        """
        if self.verbose:
            print(f"Saving degree distributions to directory: {output_dir}")

        adj_matrix = self.adj_matrix
        sns.set(style="whitegrid")

        if not self.weighted and not self.directed:
            degrees = jnp.sum(adj_matrix != 0, axis=1)
            sns.histplot(degrees, kde=True)
            plt.title(f'{title} Degree Distribution (Unweighted, Undirected)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/degree_distribution_unweighted_undirected.png')
            plt.close()
            sns.scatterplot(x=range(len(degrees)), y=degrees, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} Degree Scatter Plot (Unweighted, Undirected)', fontsize=14)
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.savefig(f'{output_dir}/degree_scatter_plot_unweighted_undirected.png')
            plt.close()
        elif self.weighted and not self.directed:
            degrees = jnp.sum(adj_matrix != 0, axis=1)
            degree_weights = jnp.sum(adj_matrix, axis=1)
            
            # Degree Distribution
            sns.histplot(degrees, kde=True)
            plt.title(f'{title} Degree Distribution (Weighted, Undirected)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/degree_distribution_weighted_undirected.png')
            plt.close()
            
            # Degree Weight Distribution
            sns.histplot(degree_weights, kde=True)
            plt.title(f'{title} Degree Weight Distribution (Weighted, Undirected)')
            plt.xlabel('Degree Weight')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/degree_weight_distribution_weighted_undirected.png')
            plt.close()
            
            # Degree vs Degree Weight Scatter Plot
            sns.scatterplot(x=degrees, y=degree_weights, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} Degree vs Degree Weight Scatter Plot (Weighted, Undirected)', fontsize=14)
            plt.xlabel('Degree')
            plt.ylabel('Degree Weight')
            plt.savefig(f'{output_dir}/degree_vs_weight_scatter_plot_weighted_undirected.png')
            plt.close()
        elif not self.weighted and self.directed:
            in_degrees = jnp.sum(adj_matrix != 0, axis=0)
            out_degrees = jnp.sum(adj_matrix != 0, axis=1)
            
            # In-Degree Distribution
            x, y = self._ccdf(in_degrees)
            plt.plot(x, y, marker='o', linestyle='none')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'{title} In-Degree Distribution (Unweighted, Directed)')
            plt.xlabel('In-Degree')
            plt.ylabel('CCDF')
            plt.savefig(f'{output_dir}/in_degree_distribution_unweighted_directed.png')
            plt.close()
            
            # Out-Degree Distribution
            x, y = self._ccdf(out_degrees)
            plt.plot(x, y, marker='o', linestyle='none')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'{title} Out-Degree Distribution (Unweighted, Directed)')
            plt.xlabel('Out-Degree')
            plt.ylabel('CCDF')
            plt.savefig(f'{output_dir}/out_degree_distribution_unweighted_directed.png')
            plt.close()
            
            # In-Degree vs Out-Degree Scatter Plot
            sns.scatterplot(x=in_degrees, y=out_degrees, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} In-Degree vs Out-Degree Scatter Plot (Unweighted, Directed)', fontsize=14)
            plt.xlabel('In-Degree')
            plt.ylabel('Out-Degree')
            if loglog:
                plt.xscale('log')
                plt.yscale('log')
            plt.savefig(f'{output_dir}/in_out_degree_scatter_plot_unweighted_directed.png')
            plt.close()
        elif self.weighted and self.directed:
            in_degrees = jnp.sum(adj_matrix != 0, axis=0)
            out_degrees = jnp.sum(adj_matrix != 0, axis=1)
            in_strengths = jnp.sum(adj_matrix, axis=0)
            out_strengths = jnp.sum(adj_matrix, axis=1)

            # In-Degree Distribution
            x, y = self._ccdf(in_degrees)
            plt.plot(x, y, marker='o', linestyle='none')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'{title} In-Degree Distribution (Weighted, Directed)')
            plt.xlabel('In-Degree')
            plt.ylabel('CCDF')
            plt.savefig(f'{output_dir}/in_degree_distribution_weighted_directed.png')
            plt.close()

            # Out-Degree Distribution
            x, y = self._ccdf(out_degrees)
            plt.plot(x, y, marker='o', linestyle='none')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'{title} Out-Degree Distribution (Weighted, Directed)')
            plt.xlabel('Out-Degree')
            plt.ylabel('CCDF')
            plt.savefig(f'{output_dir}/out_degree_distribution_weighted_directed.png')
            plt.close()

            # In-Strength Distribution
            sns.histplot(in_strengths, kde=True, color='green', label='In-Strength')
            plt.title(f'{title} In-Strength Distribution (Weighted, Directed)')
            plt.xlabel('In-Strength')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/in_strength_distribution_weighted_directed.png')
            plt.close()

            # Out-Strength Distribution
            sns.histplot(out_strengths, kde=True, color='red', label='Out-Strength')
            plt.title(f'{title} Out-Strength Distribution (Weighted, Directed)')
            plt.xlabel('Out-Strength')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/out_strength_distribution_weighted_directed.png')
            plt.close()

            # In-Degree vs In-Strength Scatter Plot
            sns.scatterplot(x=in_degrees, y=in_strengths, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} In-Degree vs In-Strength Scatter Plot (Weighted, Directed)', fontsize=14)
            plt.xlabel('In-Degree')
            plt.ylabel('In-Strength')
            if loglog:
                plt.xscale('log')
                plt.yscale('log')
            plt.savefig(f'{output_dir}/in_degree_in_strength_scatter_plot_weighted_directed.png')
            plt.close()

            # Out-Degree vs Out-Strength Scatter Plot
            sns.scatterplot(x=out_degrees, y=out_strengths, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} Out-Degree vs Out-Strength Scatter Plot (Weighted, Directed)', fontsize=14)
            plt.xlabel('Out-Degree')
            plt.ylabel('Out-Strength')
            if loglog:
                plt.xscale('log')
                plt.yscale('log')
            plt.savefig(f'{output_dir}/out_degree_out_strength_scatter_plot_weighted_directed.png')
            plt.close()

            # In-Degree vs Out-Degree Scatter Plot
            sns.scatterplot(x=in_degrees, y=out_degrees, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} In-Degree vs Out-Degree Scatter Plot (Weighted, Directed)', fontsize=14)
            plt.xlabel('In-Degree')
            plt.ylabel('Out-Degree')
            if loglog:
                plt.xscale('log')
                plt.yscale('log')
            plt.savefig(f'{output_dir}/in_out_degree_scatter_plot_weighted_directed.png')
            plt.close()

            # In-Strength vs Out-Strength Scatter Plot
            sns.scatterplot(x=in_strengths, y=out_strengths, palette='viridis', s=50, alpha=0.7, edgecolor='k')
            plt.title(f'{title} In-Strength vs Out-Strength Scatter Plot (Weighted, Directed)', fontsize=14)
            plt.xlabel('In-Strength')
            plt.ylabel('Out-Strength')
            if loglog:
                plt.xscale('log')
                plt.yscale('log')
            plt.savefig(f'{output_dir}/in_out_strength_scatter_plot_weighted_directed.png')
            plt.close()
            
        if self.verbose:
            print("Degree distributions saved.")
