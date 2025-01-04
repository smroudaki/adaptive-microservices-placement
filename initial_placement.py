from logger_setup import logger
from microservices.type import MicroserviceNodeType
from network_topology.type import NetworkNodeType


class InitialPlacementManager:
    def __init__(self, microservice_manager, network_manager):
        """
        Initialize the InitialPlacementManager with microservice and network managers.

        Args:
            microservice_manager: Manager for handling microservice operations.
            network_manager: Manager for handling network operations.
        """
        self.microservice_manager = microservice_manager
        self.network_manager = network_manager

    def place_microservices(self, groups):
        """
        Place microservices based on their groups, types, and available resources on the network.

        Args:
            groups (dict): Dictionary where keys are group IDs and values are lists of microservice IDs.

        Returns:
            dict: Placement result with microservice types as keys and lists of tuples (microservice ID, node) as values.
        """
        logger.info("Starting initial placement...")

        # Initialize placement result dictionary
        placement_result = {
            MicroserviceNodeType.SOURCE.value: [],
            MicroserviceNodeType.MODULE.value: [],
            MicroserviceNodeType.SINK.value: [],
        }
        group_first_placements = {}

        # Iterate through each group and place microservices
        for group_id, group_microservices in groups.items():
            logger.info(f"Placing microservices for group {group_id}.")

            for idx, microservice_id in enumerate(group_microservices):
                microservice = self.microservice_manager.graph.nodes[microservice_id]

                if idx == 0:
                    # Place the first microservice in the group
                    if group_id == 0:
                        placed_node = self._place_initial_microservice(microservice)
                    else:
                        # Place the first microservice of subsequent groups near the first of Group 0
                        reference_node = group_first_placements[0]
                        placed_node = self._place_microservice_nearby(
                            microservice, reference_node
                        )

                    if placed_node:
                        group_first_placements[group_id] = placed_node
                else:
                    # Place subsequent microservices in the group near their group's first microservice
                    reference_node = group_first_placements[group_id]
                    placed_node = self._place_microservice_nearby(
                        microservice, reference_node
                    )

                if placed_node:
                    placement_result[microservice["type"]].append(
                        (microservice["id"], placed_node)
                    )

        logger.info("Initial placement completed.")
        return placement_result

    def _place_initial_microservice(self, microservice):
        """
        Place the first microservice in its designated layer.

        Args:
            microservice (dict): Microservice to be placed.

        Returns:
            str: Node where the microservice is placed, or None if placement fails.
        """
        designated_layers = self._get_designated_layers(microservice["type"])
        microservice_name = microservice["name"]
        logger.info(f"Placing initial microservice {microservice_name}.")

        # Find nodes in the designated layers and check for sufficient resources
        for node in self.network_manager.find_nodes_by_layer(designated_layers):
            if self._has_sufficient_resources(node, microservice):
                self._allocate_resources(node, microservice)
                logger.info(f"Initial placement of {microservice_name} on node {node}.")
                return node

        logger.warning(f"Failed to place initial microservice {microservice_name}.")
        return None

    def _place_microservice_nearby(self, microservice, reference_node):
        """
        Place a microservice near a reference node, within its designated layer.

        Args:
            microservice (dict): Microservice to be placed.
            reference_node (str): Node near which the microservice should be placed.

        Returns:
            str: Node where the microservice is placed, or None if placement fails.
        """
        designated_layers = self._get_designated_layers(microservice["type"])
        microservice_name = microservice["name"]
        logger.info(
            f"Placing microservice {microservice_name} near node {reference_node}."
        )

        # Find nearest nodes by type and check for sufficient resources
        nearest_nodes = self.network_manager.find_nearest_nodes_by_type(
            reference_node, designated_layers
        )

        for node in nearest_nodes:
            if self._has_sufficient_resources(node, microservice):
                self._allocate_resources(node, microservice)
                logger.info(f"Placed microservice {microservice_name} on node {node}.")
                return node

        logger.warning(
            f"Failed to place microservice {microservice_name} near {reference_node}."
        )
        return None

    def _get_designated_layers(self, microservice_type):
        """
        Map microservice types to designated network layers.

        Args:
            microservice_type (str): Type of the microservice.

        Returns:
            list: List of designated network layers for the microservice type.
        """
        if microservice_type == MicroserviceNodeType.SOURCE.value:
            return [NetworkNodeType.EDGE.value]
        elif microservice_type == MicroserviceNodeType.MODULE.value:
            return [NetworkNodeType.FOG.value]
        elif microservice_type == MicroserviceNodeType.SINK.value:
            return [NetworkNodeType.CLOUD.value]
        return []

    def _has_sufficient_resources(self, node, microservice):
        """
        Check if a node has sufficient resources to host a microservice.

        Args:
            node (str): Node to check.
            microservice (dict): Microservice to be placed.

        Returns:
            bool: True if the node has sufficient resources, False otherwise.
        """
        node_data = self.network_manager.graph.nodes[node]
        return node_data["CPU"] >= self._calculate_required_cpu(microservice)

    def _allocate_resources(self, node, microservice):
        """
        Deduct resources from a node after placing a microservice.

        Args:
            node (str): Node where the microservice is placed.
            microservice (dict): Microservice to be placed.
        """
        node_data = self.network_manager.graph.nodes[node]
        node_data["CPU"] -= self._calculate_required_cpu(microservice)
        logger.info(
            f"Resources allocated on node {node}: CPU={node_data['CPU']}, RAM={node_data['RAM']}, Storage={node_data['storage']}."
        )

    def _calculate_required_cpu(self, microservice):
        """
        Calculate the required CPU for a microservice based on incoming edges.

        Args:
            microservice (dict): Microservice to be placed.

        Returns:
            int: Required CPU for the microservice.
        """
        microservice_name = microservice["name"]
        required_cpu = sum(
            self.microservice_manager.graph.edges[in_node, microservice["id"]][
                "instructions"
            ]
            for in_node in self.microservice_manager.graph.predecessors(
                microservice["id"]
            )
        )
        logger.info(
            f"Required CPU for microservice {microservice_name}: {required_cpu}."
        )
        return required_cpu
