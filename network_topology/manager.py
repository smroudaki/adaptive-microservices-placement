import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
import random
from collections import defaultdict
from config import DATA_FOLDER
from logger_setup import logger
from network_topology.edge import NetworkEdge
from network_topology.node import NetworkNode
from network_topology.type import NetworkNodeType
from network_topology.utils import find_parent, union_nodes


class NetworkTopologyManager:
    def __init__(self):
        self.graph = nx.DiGraph(name="Network Topology")
        self.next_node_id = 1
        self.next_edge_id = 1
        self.type_counters = defaultdict(int)

    def _initialize_network(self, total_nodes):
        num_edge_nodes = int(total_nodes * 0.5)
        num_fog_nodes = int(total_nodes * 0.3)
        num_cloud_nodes = total_nodes - num_edge_nodes - num_fog_nodes

        edge_nodes = self._create_nodes(
            NetworkNodeType.EDGE.value,
            num_edge_nodes,
            IPT_range=(1000, 10000),
            CPU_range=(1000, 5000),
            RAM_range=(1024, 4096),
            storage_range=(4096, 20000),
        )
        fog_nodes = self._create_nodes(
            NetworkNodeType.FOG.value,
            num_fog_nodes,
            IPT_range=(10000, 100000),
            CPU_range=(5000, 50000),
            RAM_range=(4096, 16384),
            storage_range=(20000, 100000),
        )
        cloud_nodes = self._create_nodes(
            NetworkNodeType.CLOUD.value,
            num_cloud_nodes,
            IPT_range=(100000, 1000000),
            CPU_range=(50000, 500000),
            RAM_range=(16384, 65536),
            storage_range=(100000, 1000000),
        )

        self._connect_fog_to_fog(fog_nodes)
        self._connect_edge_to_fog(edge_nodes, fog_nodes)
        self._connect_fog_to_cloud(fog_nodes, cloud_nodes)

    def _create_nodes(
        self, type, count, IPT_range, CPU_range, RAM_range, storage_range
    ):
        nodes = []
        for _ in range(count):
            self.type_counters[type] += 1
            node_name = f"{type}_{self.type_counters[type]}"
            node = NetworkNode(
                type=type,
                IPT=random.randint(*IPT_range),
                CPU=random.randint(*CPU_range),
                RAM=random.randint(*RAM_range),
                storage=random.randint(*storage_range),
            )
            node.name = node_name
            node.id = self.next_node_id
            self.graph.add_node(
                node.name,
                type=node.type,
                IPT=node.IPT,
                CPU=node.CPU,
                RAM=node.RAM,
                storage=node.storage,
                id=node.id,
            )
            self.next_node_id += 1
            nodes.append(node.name)
        return nodes

    def _connect_fog_to_fog(self, fog_nodes):
        # Create a union-find (disjoint-set) structure for fog node connectivity
        parent = {node: node for node in fog_nodes}

        # First, try to connect fog nodes within the same group (avoiding isolated nodes)
        for i, node1 in enumerate(fog_nodes):
            for node2 in fog_nodes[i + 1 :]:
                # Randomly decide whether to connect the nodes
                if random.random() < 0.3:  # 30% chance to connect
                    PR = random.randint(10, 30)
                    BW = random.randint(150, 500)
                    self._add_bidirectional_edge(node1, node2, PR, BW)
                    union_nodes(node1, node2, parent)

        # After all initial connections, make sure all fog nodes are connected
        # If there are disconnected components, connect them
        for i, node1 in enumerate(fog_nodes):
            for node2 in fog_nodes[i + 1 :]:
                if find_parent(node1, parent) != find_parent(
                    node2, parent
                ):  # If they are in different components
                    PR = random.randint(10, 30)
                    BW = random.randint(150, 500)
                    self._add_bidirectional_edge(node1, node2, PR, BW)
                    union_nodes(node1, node2, parent)

    def _connect_edge_to_fog(self, edge_nodes, fog_nodes):
        for edge in edge_nodes:
            # Each edge node connects to a random fog node
            target_fog = random.choice(fog_nodes)
            PR = random.randint(5, 15)
            BW = random.randint(50, 150)
            self._add_bidirectional_edge(edge, target_fog, PR, BW)

    def _connect_fog_to_cloud(self, fog_nodes, cloud_nodes):
        for cloud in cloud_nodes:
            # Each cloud node connects to a random fog node
            target_fog = random.choice(fog_nodes)
            PR = random.randint(20, 60)
            BW = random.randint(300, 700)
            self._add_bidirectional_edge(cloud, target_fog, PR, BW)

    def _add_bidirectional_edge(self, source, target, PR, BW):
        """
        Adds a bidirectional edge between two nodes with specified PR and BW.
        """
        edge = NetworkEdge(source=source, target=target, PR=PR, BW=BW)
        edge.id = self.next_edge_id
        self.graph.add_edge(
            source, target, name=edge.name, PR=edge.PR, BW=edge.BW, id=edge.id
        )
        self.next_edge_id += 1

        # Adding reverse edge as well
        reverse_edge = NetworkEdge(source=target, target=source, PR=PR, BW=BW)
        reverse_edge.id = self.next_edge_id
        self.graph.add_edge(
            target,
            source,
            name=reverse_edge.name,
            PR=reverse_edge.PR,
            BW=reverse_edge.BW,
            id=reverse_edge.id,
        )
        self.next_edge_id += 1

    def find_nodes_by_layer(self, layers):
        """
        Retrieve all nodes belonging to specified layers.
        :param layers: List of network layers (e.g., [NetworkNodeType.EDGE, NetworkNodeType.FOG]).
        :return: List of node names in the specified layers.
        """
        return [
            node for node, data in self.graph.nodes(data=True) if data["type"] in layers
        ]

    def find_nearest_nodes_by_type(self, reference_node, valid_layers):
        """
        Retrieve the nearest nodes of specific types to a given reference node based on the 'PR' edge weights.
        :param reference_node: The reference node from which distances are calculated.
        :param valid_layers: List of valid node types (e.g., [NetworkNodeType.FOG, NetworkNodeType.CLOUD]).
        :return: Sorted list of nodes based on their distance to the reference node.
        """
        path_lengths = nx.single_source_dijkstra_path_length(
            self.graph, source=reference_node, weight="PR"
        )
        sorted_nodes = sorted(
            (
                node
                for node in self.graph.nodes
                if self.graph.nodes[node]["type"] in valid_layers
            ),
            key=lambda n: path_lengths.get(n, float("inf")),
        )
        return sorted_nodes

    def visualize(self):
        """
        Visualize the network topology with nodes and edges.
        """
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 6))

        node_labels = {
            node: f"{NetworkNodeType(data['type']).name}_{node.split('_')[1]}"
            for node, data in self.graph.nodes(data=True)
        }
        node_colors = [
            self._node_color(data["type"]) for _, data in self.graph.nodes(data=True)
        ]

        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=node_labels,
            node_size=2000,
            node_color=node_colors,
            font_size=8,
        )

        edge_labels = {
            (src, tgt): f"{data['PR']} ms, {data['BW']} Mbps"
            for src, tgt, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_color="blue"
        )

        plt.title(self.graph.graph["name"])
        plt.show()

    @staticmethod
    def _node_color(type):
        return {
            NetworkNodeType.EDGE.value: "lightblue",
            NetworkNodeType.FOG.value: "orange",
            NetworkNodeType.CLOUD.value: "lightgreen",
        }.get(type, "gray")

    def save_graph(self, random_hash=None, filename=None):
        """Save the network topology graph to a pickle file with default path."""
        if filename is None:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), f"../{DATA_FOLDER}")
            )

            os.makedirs(base_dir, exist_ok=True)

            filename = (
                f"{base_dir}/network_topology_graph_{random_hash}.pkl"
                if random_hash
                else f"{base_dir}/network_topology_graph.pkl"
            )
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:
            data = {
                "graph": self.graph,
                "next_node_id": self.next_node_id,
                "next_edge_id": self.next_edge_id,
                "type_counters": self.type_counters,
            }
            pickle.dump(data, f)

        logger.info(f"Network topology graph saved to {filename}.")

    def load_graph(self, filename):
        """Load the network topology graph and associated attributes from a pickle file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")

        with open(filename, "rb") as f:
            data = pickle.load(f)

            self.graph = data["graph"]
            self.next_node_id = data["next_node_id"]
            self.next_edge_id = data["next_edge_id"]
            self.type_counters = data["type_counters"]

        logger.info(f"Network topology graph loaded from {filename}.")
