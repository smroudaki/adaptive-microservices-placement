import numpy as np
import os
from config import RAY_FOLDER
from logger_setup import logger
from network_topology.type import NetworkNodeType
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class ResponseTimeCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        episode_response_time_history = [
            val
            for val in env._response_time_history[-env._max_steps :]
            if val != 9999999.9
        ]
        average_episode_response_time_history = np.mean(episode_response_time_history)
        episode.custom_metrics["response_time"] = average_episode_response_time_history


class Trainer:
    def __init__(self, env, env_config):
        """
        Initialize the Trainer with the environment and its configuration.

        Args:
            env (str): The environment name.
            env_config (dict): The environment configuration.
        """
        self.env = env
        self.env_config = env_config
        self.config = None
        self.trainer = None

    def setup(self):
        """
        Setup PPO trainer with the given configurations.
        """
        self.config = (
            PPOConfig()
            .training(gamma=0.9)
            .environment(
                env=self.env,
                env_config=self.env_config,
            )
            .callbacks(ResponseTimeCallback)
            .multi_agent(
                policies={
                    f"{node_type.name.lower()}_policy" for node_type in NetworkNodeType
                },
                policy_mapping_fn=lambda agent_id, *a, **kw: f"{agent_id.split('_')[0]}_policy",
            )
            .rollouts(
                num_rollout_workers=1,
            )
            .resources(
                num_gpus=1,
                num_cpus_per_worker=4,
            )
        )

    def train(self):
        """
        Train the PPO trainer for the specified number of iterations.

        Returns:
            best_checkpoint (Checkpoint): The best checkpoint after training.
        """
        tuner = tune.Tuner(
            "PPO",
            run_config=train.RunConfig(
                storage_path=os.path.join(os.getcwd(), RAY_FOLDER),
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=5,
                    checkpoint_at_end=True,
                ),
                stop={"training_iteration": 500},
            ),
            param_space=self.config,
        )

        # Fit the tuner and get the results
        results = tuner.fit()

        # Get the best result based on the episode reward mean
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        logger.info(f"Best checkpoint saved at: {best_result.path}")

        return best_result.checkpoint
