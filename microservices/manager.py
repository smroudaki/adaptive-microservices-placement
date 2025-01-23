import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import defaultdict
from microservices.type import MicroserviceNodeType


class MicroserviceManager:
    def __init__(self, graph_name="Microservices Dependency Graph"):
        """
        Initialize the MicroserviceManager with a graph name.

        Args:
            graph_name (str): The name of the graph.
        """
        self.graph = None
        self._graph_name = graph_name
        self._type_counters = defaultdict(int)

    def create_gn_graph(self, n):
        """
        Create a GN graph and modify attributes based on indegrees and outdegrees.

        Args:
            n (int): Number of nodes in the graph.
        """
        # Create a GN graph and reverse it
        self.graph = nx.gn_graph(n - 1).reverse()

        # Relabel nodes to start from 1
        nx.relabel_nodes(self.graph, lambda x: x + 1, copy=False)

        # Add a source node (0) and an edge from 0 to 1
        self.graph.add_node(0)
        self.graph.add_edge(0, 1)

        # Set the graph name
        self.graph.name = self._graph_name

        # Generate attributes for nodes
        node_attributes = {0: self._generate_source_attributes(0)}
        for id, degree in self.graph.out_degree():
            if id == 0:
                continue
            elif degree == 0:
                node_attributes[id] = self._generate_sink_attributes(id)
                self._type_counters[MicroserviceNodeType.SINK.value] += 1
            else:
                node_attributes[id] = self._generate_module_attributes(id)
                self._type_counters[MicroserviceNodeType.MODULE.value] += 1
        nx.set_node_attributes(self.graph, node_attributes)

        # Generate attributes for edges
        edge_attributes = {}
        edge_id = 0
        for edge in reversed(list(self.graph.edges)):
            source_node_type = self.graph.nodes[edge[0]].get("type")
            target_node_type = self.graph.nodes[edge[1]].get("type")

            # Set bytes value based on target node type
            if target_node_type == MicroserviceNodeType.MODULE.value:
                bytes_value = random.randint(1000, 5000)
                memory_value = random.randint(256, 2048)
                storage_value = random.randint(10, 100)
            elif target_node_type == MicroserviceNodeType.SINK.value:
                bytes_value = random.randint(5000, 20000)
                memory_value = random.randint(512, 4096)
                storage_value = random.randint(50, 500)

            # Set instructions value based on source node type
            if source_node_type == MicroserviceNodeType.SOURCE.value:
                instructions_value = random.randint(100, 1000)
            elif source_node_type == MicroserviceNodeType.MODULE.value:
                instructions_value = random.randint(1000, 10000)

            # Set edge attributes
            edge_attributes[edge] = {
                "source": edge[0],
                "target": edge[1],
                "id": edge_id,
                "name": f"MESSAGE_{edge_id}",
                "bytes": bytes_value,
                "instructions": instructions_value,
                "memory": memory_value,
                "storage": storage_value,
            }
            edge_id += 1
        nx.set_edge_attributes(self.graph, edge_attributes)

    def _generate_source_attributes(self, id):
        """
        Generate attributes for a source node.

        Args:
            id (int): Node ID.

        Returns:
            dict: Attributes for the source node.
        """
        return {
            "id": id,
            "name": f"{MicroserviceNodeType.SOURCE.name}_{self._type_counters[MicroserviceNodeType.SOURCE.value]}",
            "type": MicroserviceNodeType.SOURCE.value,
        }

    def _generate_module_attributes(self, id):
        """
        Generate attributes for a module node.

        Args:
            id (int): Node ID.

        Returns:
            dict: Attributes for the module node.
        """
        return {
            "id": id,
            "name": f"{MicroserviceNodeType.MODULE.name}_{self._type_counters[MicroserviceNodeType.MODULE.value]}",
            "type": MicroserviceNodeType.MODULE.value,
        }

    def _generate_sink_attributes(self, id):
        """
        Generate attributes for a sink node.

        Args:
            id (int): Node ID.

        Returns:
            dict: Attributes for the sink node.
        """
        return {
            "id": id,
            "name": f"{MicroserviceNodeType.SINK.name}_{self._type_counters[MicroserviceNodeType.SINK.value]}",
            "type": MicroserviceNodeType.SINK.value,
        }

    def graph_partitioning(self):
        """
        Partition the microservices graph (`self.graph`) into communities based on dependencies.

        The method uses the Louvain algorithm to detect communities, considering edge weights
        calculated from the `bytes` and `instructions` attributes on the graph's edges.
        Larger edge weights indicate stronger dependencies, influencing the partitioning.

        Returns:
            dict: A dictionary where keys are group IDs (int) and values are lists of node names
                belonging to each group.
        """
        # Convert the directed graph to an undirected graph for community detection
        undirected_graph = self.graph.to_undirected()

        # Assign edge weights based on bytes and instructions
        for _, _, data in undirected_graph.edges(data=True):
            # Weight is a combination of bytes (data volume) and instructions (computational cost)
            data["weight"] = data["bytes"] + 0.1 * data["instructions"]

        # Perform Louvain community detection with a configurable resolution parameter
        communities = nx.community.louvain_communities(
            undirected_graph,
            weight="weight",  # Use the calculated weights for partitioning
            resolution=1.0,  # Adjust to control community size (default: 1.0)
        )

        # Convert communities (list of sets) into a dictionary format with group IDs
        groups = {group_id: list(group) for group_id, group in enumerate(communities)}

        # Sort groups by group ID and return as a sorted dictionary
        sorted_groups = {key: groups[key] for key in sorted(groups.keys())}

        return sorted_groups

    def visualize(self, groups=None):
        """
        Visualize the microservice dependency graph.

        Args:
            groups (dict, optional): A dictionary where keys are group IDs (int) and values are lists of node names
                belonging to each group. If provided, nodes will be colored based on their group.
        """
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 8))

        # Assign colors to nodes based on groups if provided
        node_colors = []
        if groups:
            unique_groups = sorted(groups.keys())
            color_map = {
                group_id: plt.cm.tab20(i / len(unique_groups))
                for i, group_id in enumerate(unique_groups)
            }
            for node in self.graph.nodes:
                node_color = next(
                    (
                        color_map[group_id]
                        for group_id, nodes in groups.items()
                        if node in nodes
                    ),
                    "lightgrey",
                )
                node_colors.append(node_color)
        else:
            node_colors = "lightgrey"

        # Node labels based on type and type-specific counter
        node_labels = {node: data["name"] for node, data in self.graph.nodes(data=True)}

        # Draw graph with node and edge labels
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
            (src, tgt): f"{data['instructions']} instr, {data['bytes']} B"
            for src, tgt, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_color="red"
        )

        plt.title(f"{self.graph.graph['name']} - Group Visualization")
        plt.show()
