import torch
import pickle

import torch
from torch_geometric.data import Data, DataLoader


class LinkPredictionDataset():
    def __init__(self, nx_graph):
        """
        Args:
            nx_graph (NetworkX.Graph): The NetworkX graph.
            test_size (float): Fraction of edges to put in the test set.
            num_negative_samples (int): Number of negative samples to generate. If None, use the same number as positive samples.
            batch_size (int): Batch size for DataLoader.
        """
        self.nx_graph = nx_graph
        self.edge_type_mapping = {'e-p': 0, 'e-m': 1, 'e-d': 2, 'e-me': 3, 'd-e': 2}
        self.node_type_mapping = {'e': 0, 'p': 1, 'm': 2, 'd': 3, 'me': 4}
        # Convert the NetworkX graph to PyTorch Geometric format
        self.nodes, self.edges, self.node_types, self.edge_types, self.edge_values, node_mappings, reversed_mapping, node_type_mapping = self._convert_to_pyg()
        self.node_mappings = node_mappings
        self.reverse_node_mappings = reversed_mapping
        self.node_type_mapping = node_type_mapping
        self.data = Data(
            x=self.nodes.view(-1, 1),
            node_type=self.node_types.view(-1, 1),
            edge_index=self.edges.t().contiguous(),
            edge_type=self.edge_types.view(-1, 1),
            edge_attr=self.edge_values.view(-1, 1)
        )
        # has no features for the nodes, did not include the edges

    def _convert_to_pyg(self):

        node_mapping = {}
        nodes = []
        edges = []
        node_types = []
        edge_values = []
        edge_types = []
        node_type_mapping = {}
        for node in self.nx_graph.nodes():
            node_idx = len(nodes)
            nodes.append(node_idx)
            node_mapping[node] = node_idx

            node_type = self.node_type_mapping[node.split('_')[0]]
            node_types.append(node_type)
            node_type_mapping[node_idx] = node_type

        for edge in self.nx_graph.edges():
            src, dst = edge
            edges.append((node_mapping[src], node_mapping[dst]))
            edge_values.append(self.nx_graph[src][dst].get('value', 0))
            edge_type = self.edge_type_mapping[self.nx_graph[src][dst]['type']]
            edge_types.append(edge_type)

            edges.append((node_mapping[dst], node_mapping[src]))
            edge_values.append(self.nx_graph[src][dst].get('value', 0))
            edge_type = self.edge_type_mapping[self.nx_graph[src][dst]['type']]
            edge_types.append(edge_type)

        reverse_node_mappings = {}
        for ori_node, node_idx in node_mapping.items():
            reverse_node_mappings[node_idx] = ori_node
        return (
            torch.tensor(nodes, dtype=torch.long),
            torch.tensor(edges, dtype=torch.long),
            torch.tensor(node_types, dtype=torch.long),
            torch.tensor(edge_types, dtype=torch.long),
            torch.tensor(edge_values, dtype=torch.float),
            node_mapping,
            reverse_node_mappings,
            node_type_mapping)
