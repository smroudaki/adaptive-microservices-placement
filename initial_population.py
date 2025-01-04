from yafs.distribution import deterministicDistributionStartPoint
from microservices.type import MicroserviceNodeType


class InitialPopulationManager:
    def __init__(self, microservice_manager, placement):
        """
        Initializes the InitialPopulationManager with a microservice manager and placement.

        Args:
            microservice_manager: The manager handling microservice operations.
            placement: The placement of microservices on nodes.
        """
        self.microservice_manager = microservice_manager
        self.placement = placement

    def create_population(self):
        """
        Generate a population for an application that includes sources with their connected resources
        from placement results and sinks with assigned resources.

        Returns:
            dict: A dictionary representing the population of the application.
        """
        population = {}
        app_name = self.microservice_manager.graph.name
        population[app_name] = {"sources": [], "sinks": []}

        # Constants for source configuration
        start_time = 0  # Start time for the source messages
        message_interval = 5000  # Interval between messages
        number_of_requests = 1  # Number of requests to be sent

        # Process SOURCE microservices
        for ms_id, allocated_node in self.placement[MicroserviceNodeType.SOURCE.value]:
            # Iterate over all target nodes connected to this SOURCE
            target_nodes = list(self.microservice_manager.graph.successors(ms_id))

            for target in target_nodes:
                # Retrieve the edge (message) name connecting source to target
                if self.microservice_manager.graph.has_edge(ms_id, target):
                    edge_data = self.microservice_manager.graph.get_edge_data(
                        ms_id, target
                    )
                    message_name = edge_data["name"]

                    # Add source entry to population using the edge (message) name
                    population[app_name]["sources"].append(
                        (
                            message_name,
                            allocated_node,
                            number_of_requests,
                            deterministicDistributionStartPoint(
                                name="Deterministic",
                                start=start_time,
                                time=message_interval,
                            ),
                        )
                    )

        # Process SINK microservices
        for sink_name, allocated_node in self.placement[
            MicroserviceNodeType.SINK.value
        ]:
            # Add sink entry to population
            population[app_name]["sinks"].append((allocated_node, sink_name))

        return population
