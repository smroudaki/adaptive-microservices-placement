import os
import pickle
from config import DATA_FOLDER
from logger_setup import logger


def save_setup(
    microservice_manager,
    network_manager,
    groups,
    initial_placement,
    initial_population,
    random_hash=None,
    file_path=None,
):
    """
    Save to a pickle file with default path.

    Args:
        random_hash (str, optional): A random hash to include in the file path.
        file_path (str, optional): The file path to save the setup to. If not provided, a default path is used.
    """
    if file_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_FOLDER))

        os.makedirs(base_dir, exist_ok=True)

        file_path = (
            f"{base_dir}/setup_{random_hash}.pkl"
            if random_hash
            else f"{base_dir}/setup.pkl"
        )
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        data = {
            "microservice_manager": microservice_manager,
            "network_manager": network_manager,
            "groups": groups,
            "initial_placement": initial_placement,
            "initial_population": initial_population,
        }

        pickle.dump(data, f)

    logger.info(f"Saved setup to {file_path}.")


def load_setup(file_path):
    """
    Load from a pickle file.

    Args:
        file_path (str): The file path to load the setup from.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

        microservice_manager = data["microservice_manager"]
        network_manager = data["network_manager"]
        groups = data["groups"]
        initial_placement = data["initial_placement"]
        initial_population = data["initial_population"]

    logger.info(f"Loaded setup from {file_path}.")

    return (
        microservice_manager,
        network_manager,
        groups,
        initial_placement,
        initial_population,
    )
