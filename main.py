import argparse
import os
import random
import ray
import uuid
import warnings
from adaptive_microservices_placement_env import AdaptiveMicroservicesPlacementEnv
from config import (
    MICROSERVICE_NODES_MIN,
    MICROSERVICE_NODES_MAX,
    NETWORK_NODES_MIN,
    NETWORK_NODES_MAX,
)
from initial_placement import InitialPlacementManager
from microservices.manager import MicroserviceManager
from network_topology.manager import NetworkTopologyManager
from initial_population import InitialPopulationManager
from setup_manager import save_setup, load_setup
from trainer import Trainer
from logger_setup import logger

os.environ["RAY_DEDUP_LOGS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    parser = argparse.ArgumentParser(description="Load setups.")
    parser.add_argument(
        "--setup-path",
        type=str,
        default=None,
        help="Path to the setup (pickle file).",
    )
    args = parser.parse_args()

    if args.setup_path and os.path.exists(args.setup_path):
        logger.info("Loading setup...")
        (
            microservice_manager,
            network_manager,
            groups,
            initial_placement,
            initial_population,
        ) = load_setup(args.setup_path)

        logger.info("Microservices groups: %s", groups)
        logger.info("Initial microservices placement: %s", initial_placement)
        logger.info("Population: %s", initial_population)

        microservice_manager.visualize(groups)
        network_manager.visualize()
    else:
        random_hash = uuid.uuid4().hex[:6]

        logger.info("Creating new microservices graph...")
        microsevices_nodes_num = random.randint(
            MICROSERVICE_NODES_MIN, MICROSERVICE_NODES_MAX
        )
        microsevices_nodes_num = MICROSERVICE_NODES_MIN
        logger.info("Total microservices graph nodes: %s", microsevices_nodes_num)
        microservice_manager = MicroserviceManager()
        microservice_manager.create_gn_graph(microsevices_nodes_num)

        logger.info("Creating new network topology graph...")
        network_nodes_num = random.randint(NETWORK_NODES_MIN, NETWORK_NODES_MAX)
        network_nodes_num = NETWORK_NODES_MIN
        logger.info("Total network topology graph nodes: %s", network_nodes_num)
        network_manager = NetworkTopologyManager()
        network_manager._initialize_network(network_nodes_num)

        groups = microservice_manager.graph_partitioning()
        logger.info("Microservices groups: %s", groups)

        placement_manager = InitialPlacementManager(
            microservice_manager, network_manager
        )
        initial_placement = placement_manager.place_microservices(groups)
        logger.info("Initial microservices placement: %s", initial_placement)

        population_manager = InitialPopulationManager(
            microservice_manager, initial_placement
        )
        initial_population = population_manager.create_population()
        logger.info("Population: %s", initial_population)

        save_setup(
            microservice_manager,
            network_manager,
            groups,
            initial_placement,
            initial_population,
            random_hash,
        )

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

    logger.info("Best model checkpoint: %s", best_model_checkpoint)


if __name__ == "__main__":
    main()
