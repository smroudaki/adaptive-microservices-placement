import math
import numpy as np
from config import (
    CHANGE_LOCATION_INTERVAL,
    CHANGE_REQUESTS_INTERVAL,
    CHANGE_DYNAMICS_CYCLES,
    YAFS_RUN_UNTIL,
)
from gymnasium import spaces
from logger_setup import logger
from microservices.type import MicroserviceNodeType
from network_topology.type import NetworkNodeType
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# from ray.tune.registry import register_env
from utils.yafs.converter import dag_to_yafs_application, network_to_yafs_topology
from utils.yafs.leakers import RunTimeLeaker
from utils.yafs.placements import MappedPlacement
from utils.yafs.populations import MappedPopulationImproved
from yafs.core import Sim
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.distribution import deterministicDistributionStartPoint


class AdaptiveMicroservicesPlacementEnv(MultiAgentEnv):
    def __init__(self, env_config):
        """
        Initialize the AdaptiveMicroservicesPlacementEnv environment.

        Args:
            env_config (dict): Configuration dictionary containing microservice manager, network manager,
                               initial placement, and initial population.
        """
        super().__init__()

        # Initialize environment configuration
        self._microservice_manager = env_config["microservice_manager"]
        self._network_manager = env_config["network_manager"]
        self._initial_placement = env_config["initial_placement"]
        self._initial_population = env_config["initial_population"]

        self._agent_ids = [
            node_type.name.lower() + "_agent" for node_type in NetworkNodeType
        ]

        # Calculate the least common multiple (LCM) of change intervals to determine max steps
        lcm = abs(CHANGE_LOCATION_INTERVAL * CHANGE_REQUESTS_INTERVAL) // math.gcd(
            CHANGE_LOCATION_INTERVAL, CHANGE_REQUESTS_INTERVAL
        )
        self._max_steps = lcm * CHANGE_DYNAMICS_CYCLES

        self._num_placement_microservices = len(
            self._initial_placement[MicroserviceNodeType.MODULE.value]
        ) + len(self._initial_placement[MicroserviceNodeType.SINK.value])

        # Define per-agent observation and action spaces
        self._agent_observation_space = spaces.Dict(
            {
                "current_response_time": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "placement_map": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        self._num_placement_microservices,
                        len(NetworkNodeType.__members__),
                    ),
                    dtype=np.int32,
                ),
                "available_resources": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(len(self._network_manager.graph.nodes), 1),
                    dtype=np.float32,
                ),
            }
        )
        self._agent_action_space = spaces.Tuple(
            (
                spaces.Discrete(
                    self._num_placement_microservices + 1
                ),  # Select microservice (0 = no action)
                spaces.Discrete(len(NetworkNodeType.__members__)),  # Target layer index
            )
        )

        # Assign spaces to all agents
        self.observation_space = spaces.Dict(
            {agent: self._agent_observation_space for agent in self._agent_ids}
        )
        self.action_space = spaces.Dict(
            {agent: self._agent_action_space for agent in self._agent_ids}
        )

        self._step = 0
        self._enhanced_placement_map = self._initialize_enhanced_placement_map()
        self._response_time_history = [self._run_yafs_simulation()]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return initial observations for all agents.
        """
        obs = self._get_observations()
        infos = {agent: {} for agent in self._agent_ids}

        return obs, infos

    def step(self, action_dict):
        """
        Apply actions, run simulation, and return new observations, rewards, and dones.

        Args:
            action_dict (dict): Dictionary of actions for each agent.

        Returns:
            tuple: New observations, rewards, terminated flags, truncated flags, and info dictionary for all agents.
        """
        self._step += 1
        logger.info(f"Step: {self._step}")
        logger.info(f"Action: {action_dict}")

        self._handle_dynamic_changes()

        self.rewards = {}

        for agent_id, action in action_dict.items():
            ms_idx, target_layer_idx = map(int, action)

            if ms_idx == 0:
                self.rewards[agent_id] = -0.1
                continue

            migration_success = self._migrate_microservice(ms_idx, target_layer_idx)

            if not migration_success:
                self.rewards[agent_id] = -1

        response_time = self._run_yafs_simulation()

        for agent in self._agent_ids:
            if agent not in self.rewards:
                self.rewards[agent] = 1 / response_time

        obs = self._get_observations()

        terminateds = {agent: False for agent in self._agent_ids} | {
            "__all__": self._step % self._max_steps == 0
        }

        truncateds = {agent: False for agent in self._agent_ids} | {"__all__": False}

        infos = {agent: {} for agent in self._agent_ids}

        self._response_time_history.append(response_time)

        self.render(mode="human")

        return obs, self.rewards, terminateds, truncateds, infos

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): Render mode.
        """
        for agent, reward in self.rewards.items():
            logger.info(f"Reward for {agent}: {reward}")
            print(f"Reward for {agent}: ", reward)
        logger.info(f"Response Time: {self._response_time_history[-1]}")
        print(f"Response Time: {self._response_time_history[-1]}")

    def _initialize_enhanced_placement_map(self):
        """
        Initialize enhanced initial placement with NODE_ID and placement index for MODULE and SINK types.

        Returns:
            dict: Enhanced initial placement dictionary.
        """
        enhanced_initial_placement = {}
        ms_idx = 1

        for ms_type, placements in self._initial_placement.items():
            enhanced_initial_placement[ms_type] = []

            for ms_id, node_name in placements:
                ms_name = self._microservice_manager.graph.nodes[ms_id]["name"]
                node_id = self._network_manager.graph.nodes[node_name]["id"]

                if ms_type in [
                    MicroserviceNodeType.MODULE.value,
                    MicroserviceNodeType.SINK.value,
                ]:
                    enhanced_initial_placement[ms_type].append(
                        (ms_name, node_name, ms_id, node_id, ms_idx)
                    )
                    ms_idx += 1
                else:
                    enhanced_initial_placement[ms_type].append(
                        (ms_name, node_name, ms_id, node_id)
                    )

        logger.info("Enhanced initial placement initialized.")
        return enhanced_initial_placement

    def _to_lite_placement_map(self, enhanced_placement_map):
        """
        Initialize the placement map from the enhanced initial placement, including placement indices.

        Returns:
            np.ndarray: Placement map array.
        """
        placement_map = np.zeros(
            (self._num_placement_microservices, len(NetworkNodeType.__members__)),
            dtype=np.int32,
        )

        for ms_type, placements in enhanced_placement_map.items():
            for placement in placements:
                if ms_type in [
                    MicroserviceNodeType.MODULE.value,
                    MicroserviceNodeType.SINK.value,
                ]:
                    _, node_name, _, _, ms_idx = placement
                    layer_idx = self._network_manager.graph.nodes[node_name]["type"]
                    placement_map[ms_idx - 1, layer_idx] = 1

        return placement_map

    def _migrate_microservice(self, ms_idx, target_layer_idx):
        """
        Perform a migration of the specified microservice to a target layer.

        Args:
            ms_idx (int): The placement index of the microservice to migrate.
            target_layer_idx (int): The target layer index (0=Edge, 1=Fog, 2=Cloud).

        Returns:
            bool: Whether the migration was successful.
        """
        placement_info = None

        # Locate the microservice placement information
        for ms_type, placements in self._enhanced_placement_map.items():
            if ms_type in [
                MicroserviceNodeType.MODULE.value,
                MicroserviceNodeType.SINK.value,
            ]:
                for p in placements:
                    if p[-1] == ms_idx:
                        placement_info = p
                        break
            if placement_info:
                break

        if not placement_info:
            logger.warning(f"No placement info found for ms_idx: {ms_idx}")
            return False

        ms_name, current_node, ms_id, _, _ = placement_info
        target_layer = [node_type.value for node_type in NetworkNodeType][
            target_layer_idx
        ]

        if self._network_manager.graph.nodes[current_node]["type"] == target_layer:
            logger.info(
                f"Microservice {ms_name} already on the target layer {target_layer}. No migration performed."
            )
            return True

        target_layer_nodes = self._network_manager.find_nearest_nodes_by_type(
            current_node, [target_layer]
        )

        for target_node in target_layer_nodes:
            if self._has_sufficient_resources(target_node, ms_id):
                self._free_resources(current_node, ms_id)
                self._allocate_resources(target_node, ms_id)
                self._enhanced_placement_map[ms_type].remove(placement_info)
                self._enhanced_placement_map[ms_type].append(
                    (
                        ms_name,
                        target_node,
                        ms_id,
                        self._network_manager.graph.nodes[target_node]["id"],
                        ms_idx,
                    )
                )

                logger.info(
                    f"Microservice {ms_name} migrated from node {current_node} to {target_node} (layer {target_layer_idx})."
                )
                return True

        logger.warning(
            f"Migration failed for microservice {ms_name} (ms_idx={ms_idx}) to layer {target_layer_idx}. No suitable node."
        )
        return False

    def _handle_dynamic_changes(self):
        """
        Handle dynamic changes to user request patterns and source microservice locations.
        """
        if self._step % CHANGE_LOCATION_INTERVAL == 0:
            self._change_source_locations()

        if self._step % CHANGE_REQUESTS_INTERVAL == 0:
            self._update_user_requests()

    def _change_source_locations(self):
        """
        Change the network location of source-type microservices to model mobility.
        """
        app_name = self._microservice_manager.graph.name
        sources = self._initial_population[app_name]["sources"]

        for idx, (
            message_name,
            current_node,
            number_of_requests,
            distribution,
        ) in enumerate(sources):
            nearest_nodes = self._network_manager.find_nearest_nodes_by_type(
                current_node, [NetworkNodeType.EDGE.value]
            )
            ms_id = next(
                (
                    u
                    for u, _, attr in self._microservice_manager.graph.edges(data=True)
                    if attr["name"] == message_name
                ),
                None,
            )

            new_node = None
            for candidate_node in nearest_nodes:
                if self._has_sufficient_resources(candidate_node, ms_id):
                    new_node = candidate_node
                    break

            if not new_node:
                continue

            self._initial_population[app_name]["sources"][idx] = (
                message_name,
                new_node,
                number_of_requests,
                distribution,
            )

            # Update enhanced initial placement
            for placement in self._enhanced_placement_map[
                MicroserviceNodeType.SOURCE.value
            ]:
                if placement[0] == ms_id:
                    self._enhanced_placement_map[
                        MicroserviceNodeType.SOURCE.value
                    ].remove(placement)
                    self._enhanced_placement_map[
                        MicroserviceNodeType.SOURCE.value
                    ].append(
                        (
                            placement[0],
                            new_node,
                            placement[2],
                            self._network_manager.graph.nodes[new_node]["id"],
                        )
                    )
                    break

        logger.info("Source locations updated.")

    def _update_user_requests(self):
        """
        Update user requests based on the population and simulate changes in request patterns.
        """
        app_name = self._microservice_manager.graph.name
        sources = self._initial_population[app_name]["sources"]
        updated_sources = []

        for source in sources:
            message_name, allocated_node, _, _ = source
            if np.random.rand() < 0.5:
                new_number_of_requests = np.random.randint(1, 5)
                new_message_interval = np.random.randint(1000, 10000)

                updated_sources.append(
                    (
                        message_name,
                        allocated_node,
                        new_number_of_requests,
                        deterministicDistributionStartPoint(
                            name="Deterministic", start=0, time=new_message_interval
                        ),
                    )
                )
            else:
                updated_sources.append(source)

        self._initial_population[app_name]["sources"] = updated_sources
        logger.info("User requests updated.")

    def _has_sufficient_resources(self, node, ms_id):
        """
        Check if a node has sufficient resources for the microservice.

        Args:
            node (str): Node name.
            ms_id (int): Microservice ID.

        Returns:
            bool: Whether the node has sufficient resources.
        """
        node_data = self._network_manager.graph.nodes[node]
        required_cpu = self._calculate_required_cpu(ms_id)
        return node_data["CPU"] >= required_cpu

    def _free_resources(self, node, ms_id):
        """
        Free resources on a node for the microservice.

        Args:
            node (str): Node name.
            ms_id (int): Microservice ID.
        """
        node_data = self._network_manager.graph.nodes[node]
        required_cpu = self._calculate_required_cpu(ms_id)
        node_data["CPU"] += required_cpu
        logger.info(
            f"Freed CPU resources on node {node}: Available CPU = {node_data['CPU']}"
        )

    def _allocate_resources(self, node, ms_id):
        """
        Allocate resources on a node for the microservice.

        Args:
            node (str): Node name.
            ms_id (int): Microservice ID.
        """
        node_data = self._network_manager.graph.nodes[node]
        required_cpu = self._calculate_required_cpu(ms_id)
        node_data["CPU"] -= required_cpu
        logger.info(
            f"Allocated CPU resources on node {node}: Remaining CPU = {node_data['CPU']}"
        )

    def _calculate_required_cpu(self, ms_id):
        """
        Calculate the CPU required for a microservice based on incoming edges.

        Args:
            ms_id (int): Microservice ID.

        Returns:
            int: Required CPU.
        """
        return sum(
            self._microservice_manager.graph.edges[in_node, ms_id]["instructions"]
            for in_node in self._microservice_manager.graph.predecessors(ms_id)
        )

    def _run_yafs_simulation(self):
        """
        Run the YAFS simulation and return the response time.

        Returns:
            float: Response time.
        """
        sim_application = dag_to_yafs_application(self._microservice_manager.graph)
        sim_topology = network_to_yafs_topology(self._network_manager.graph)
        sim_placement = MappedPlacement(
            {
                self._microservice_manager.graph.name: [
                    (ms_id, node_name)
                    for ms_type, placements in self._enhanced_placement_map.items()
                    for _, node_name, ms_id, *_ in placements
                    if ms_type == MicroserviceNodeType.MODULE.value
                ]
            },
            name="MappedPlacement",
        )
        sim_population = MappedPopulationImproved(
            self._initial_population, name="AutoGeneratedPopulation"
        )
        sim_selector = DeviceSpeedAwareRouting()

        sim = Sim(
            topology=sim_topology,
            metrics=RunTimeLeaker(),
        )
        sim.deploy_app2(sim_application, sim_placement, sim_population, sim_selector)
        response_time = sim.run(YAFS_RUN_UNTIL)

        return response_time

    def _get_observations(self):
        """
        Return the current state as observations for all agents.

        Returns:
            dict: Observations for all agents.
        """
        current_response_time = self._response_time_history[-1]
        placement_map = self._to_lite_placement_map(self._enhanced_placement_map)
        available_resources = [
            [self._network_manager.graph.nodes[node]["CPU"]]
            for node in self._network_manager.graph.nodes
        ]

        return {
            agent: {
                "current_response_time": np.array(
                    [current_response_time], dtype=np.float32
                ),
                "placement_map": placement_map,
                "available_resources": np.array(available_resources, dtype=np.float32),
            }
            for agent in self._agent_ids
        }


# def env_creator(env_config):
#     return AdaptiveMicroservicesPlacementEnv(env_config)


# register_env("adaptive_microservices_placement_env", env_creator)
