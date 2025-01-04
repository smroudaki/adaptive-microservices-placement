from yafs.population import Population


class CompatiblePopulation(Population):
    """
    A modified version of YAFS example population kept here for future reference.
    """

    def initial_allocation(self, sim, app_name):
        """
        Allocates initial sinks and sources for the simulation.

        Parameters:
        sim (Simulation): The simulation object.
        app_name (str): The name of the application to deploy.

        Returns:
        None
        """
        # Deploy sinks based on the sink control configuration
        for ctrl in self.sink_control:
            sim.deploy_sink(app_name, node=ctrl["id"], module=ctrl["module"])

        # Deploy sources based on the source control configuration
        for ctrl in self.src_control:
            for _ in range(ctrl["number"]):
                sim.deploy_source(
                    app_name,
                    id_node=ctrl["id"],
                    msg=ctrl["message"],
                    distribution=ctrl["distribution"],
                )


class MappedPopulation(Population):
    """
    A Static population assigning sources and sinks based on a global dictionary.
    """

    def __init__(self, population_map, **kwargs):
        """
        Initializes the MappedPopulation with a population map.

        Parameters:
        population_map (dict): A dictionary containing the mapping of sources and sinks.
        kwargs: Additional keyword arguments.

        Returns:
        None
        """
        self.population_map = population_map
        super().__init__(**kwargs)

    def initial_allocation(self, sim, app_name):
        """
        Allocates initial sinks and sources for the simulation based on the population map.

        Parameters:
        sim (Simulation): The simulation object.
        app_name (str): The name of the application to deploy.

        Returns:
        None
        """
        # A dictionary of actual message objects originating from source modules
        source_messages = sim.apps[app_name].messages

        # Break assignment information into source and sink information
        source_info, sink_info = self.population_map[app_name]

        # Deploy sources based on the source information
        for message in source_messages:
            for _ in range(source_info[1]):
                sim.deploy_source(
                    app_name,
                    id_node=source_info[0],
                    msg=source_messages[message],
                    distribution=source_info[2],
                )

        # Deploy sinks based on the sink information
        for device_id, module_name in sink_info:
            sim.deploy_sink(app_name, node=device_id, module=module_name)


class MappedPopulationImproved(Population):
    """
    A Static population assigning sources and sinks based on a global dictionary.
    """

    def __init__(self, population_map, **kwargs):
        """
        Initializes the MappedPopulationImproved with a population map.

        Parameters:
        population_map (dict): A dictionary containing the mapping of sources and sinks.
        kwargs: Additional keyword arguments.

        Returns:
        None
        """
        self.population_map = population_map
        super().__init__(**kwargs)

    def initial_allocation(self, sim, app_name):
        """
        Allocates initial sinks and sources for the simulation based on the population map.

        Parameters:
        sim (Simulation): The simulation object.
        app_name (str): The name of the application to deploy.

        Returns:
        None
        """
        # A dictionary of actual message objects originating from source modules
        source_messages = sim.apps[app_name].messages

        # Iterate over each application in the population map
        for app_name, app_data in self.population_map.items():
            # Deploy sources based on the source information
            for message, device_id, message_count, distribution in app_data["sources"]:
                for _ in range(message_count):
                    sim.deploy_source(
                        app_name,
                        id_node=device_id,
                        msg=source_messages[message],
                        distribution=distribution,
                    )

            # Deploy sinks based on the sink information
            for device_id, module_name in app_data["sinks"]:
                sim.deploy_sink(app_name, node=device_id, module=module_name)
