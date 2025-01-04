from yafs.placement import Placement


class MappedPlacement(Placement):
    """
    A static placement class placing modules over devices based on a global dictionary, initial_map.
    """

    def __init__(self, initial_map: dict[str, list[tuple[str, int]]], **kwargs):
        """
        Initialize the MappedPlacement instance.

        Parameters:
        initial_map (dict): A dictionary of structure, {app_name: [(module_name, device_id), ...]}.
        kwargs (dict): Additional keyword arguments to pass to the Placement superclass.

        Example:
        initial_map = {application_0: [(Module_0, 12), (Module_1, 12), (Module_2, 6), (Module_2, 4), (Module_3, 1)]}
        placement = MappedPlacement(initial_map)
        """
        super(MappedPlacement, self).__init__(**kwargs)
        self.initial_map = initial_map

    def initial_allocation(self, sim, app_name):
        """
        Allocate initial placement of modules for a given application.

        Parameters:
        sim (Simulation): The simulation instance.
        app_name (str): The name of the application to allocate modules for.

        Returns:
        None

        Raises:
        Exception: If the application name is not found in the initial_map.
        """
        services = sim.apps[app_name].services  # Get the services of the application

        if not app_name in self.initial_map:
            raise Exception(
                f"Application {app_name} was not placed."
            )  # Raise an exception if the app is not in the initial_map

        for module, node in self.initial_map[app_name]:
            sim.deploy_module(
                app_name, module, services[module], [node]
            )  # Deploy each module to the specified node
