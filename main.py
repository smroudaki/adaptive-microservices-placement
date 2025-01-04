import argparse
import os
import ray
import uuid
import warnings
from adaptive_microservices_placement_env import AdaptiveMicroservicesPlacementEnv
from config import MICROSERVICE_NODES, NETWORK_NODES
from initial_placement import InitialPlacementManager
from microservices.manager import MicroserviceManager
from network_topology.manager import NetworkTopologyManager
from initial_population import InitialPopulationManager
from trainer import Trainer
from logger_setup import logger

os.environ["RAY_DEDUP_LOGS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    parser = argparse.ArgumentParser(
        description="Load or create microservices and network topology graphs."
    )
    parser.add_argument(
        "--microservices-graph",
        type=str,
        default=None,
        help="Path to the saved microservices graph (pickle file).",
    )
    parser.add_argument(
        "--network-topology-graph",
        type=str,
        default=None,
        help="Path to the saved network topology graph (pickle file).",
    )
    args = parser.parse_args()

    microservice_manager = MicroserviceManager()
    network_manager = NetworkTopologyManager()

    # Generate random_hash if needed (only if you want to use the default filename pattern)
    if not args.microservices_graph or not args.network_topology_graph:
        random_hash = uuid.uuid4().hex[:6]

    # Load the microservices graph if a path is provided, otherwise create new
    if args.microservices_graph and os.path.exists(args.microservices_graph):
        logger.info("Loading saved microservices graph...")
        microservice_manager.load_graph(args.microservices_graph)
    else:
        logger.info("Creating new microservices graph...")
        logger.info("Total Microservices: %s", MICROSERVICE_NODES)
        microservice_manager.create_gn_graph(MICROSERVICE_NODES)
        microservice_manager.save_graph(random_hash=random_hash)

    # Load the network topology graph if a path is provided, otherwise create new
    if args.network_topology_graph and os.path.exists(args.network_topology_graph):
        logger.info("Loading saved network topology graph...")
        network_manager.load_graph(args.network_topology_graph)
    else:
        logger.info("Creating new network topology graph...")
        network_manager._initialize_network(NETWORK_NODES)
        network_manager.save_graph(random_hash=random_hash)

    groups = microservice_manager.graph_partitioning()
    logger.info("Groups: %s", groups)

    placement_manager = InitialPlacementManager(microservice_manager, network_manager)
    initial_placement = placement_manager.place_microservices(groups)
    logger.info("Initial Placement Result: %s", initial_placement)

    population_manager = InitialPopulationManager(
        microservice_manager, initial_placement
    )
    initial_population = population_manager.create_population()
    logger.info("Population: %s", initial_population)

    microservice_manager.visualize(groups)
    network_manager.visualize()

    ray.init()
    env_config = {
        "microservice_manager": microservice_manager,
        "network_manager": network_manager,
        "initial_placement": initial_placement,
        "initial_population": initial_population,
    }
    trainer = Trainer(env=AdaptiveMicroservicesPlacementEnv, env_config=env_config)
    trainer.setup()
    best_model_checkpoint = trainer.train()

    logger.info("Best Model Checkpoint: %s", best_model_checkpoint)


if __name__ == "__main__":
    main()
