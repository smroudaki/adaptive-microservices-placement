import matplotlib.pyplot as plt
import numpy as np
import re


def extract_response_times(file_path):
    response_times = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"Response Time:\s*([\d.]+)", line)
            if match:
                response_times.append(float(match.group(1)))
    return response_times


def plot_response_times(response_times, step=100000):
    mean_response_times = [
        np.mean(response_times[i : i + step])
        for i in range(0, len(response_times), step)
    ]
    plt.plot(mean_response_times)
    plt.xlabel("Step")
    plt.ylabel("Response Time")
    plt.show()


def extract_rewards(file_path):
    """
    Extract reward values from log file for each agent.

    Args:
        file_path (str): Path to the log file.

    Returns:
        dict: {'edge_agent': [...], 'fog_agent': [...], 'cloud_agent': [...]}
    """
    edge_rewards, fog_rewards, cloud_rewards = [], [], []
    with open(file_path, "r") as file:
        for line in file:
            edge_match = re.search(r"Reward for edge_agent:\s+(.+)", line)
            if edge_match:
                edge_rewards.append(float(edge_match.group(1)))
            fog_match = re.search(r"Reward for fog_agent:\s+(.+)", line)
            if fog_match:
                fog_rewards.append(float(fog_match.group(1)))
            cloud_match = re.search(r"Reward for cloud_agent:\s+(.+)", line)
            if cloud_match:
                cloud_rewards.append(float(cloud_match.group(1)))
    return {
        "edge_agent": edge_rewards,
        "fog_agent": fog_rewards,
        "cloud_agent": cloud_rewards,
    }


def plot_rewards(rewards, step=100000):
    """
    Plot mean reward for each agent in steps, plus overall mean.

    Args:
        rewards (dict): Dictionary of agent -> list of rewards.
        step (int, optional): Step size for averaging. Defaults to 100000.

    Returns:
        None
    """
    agent_means = {}
    for agent, vals in rewards.items():
        agent_means[agent] = [
            np.mean(vals[i : i + step]) for i in range(0, len(vals), step)
        ]

    max_length = max(len(m) for m in agent_means.values())
    overall_mean = []
    for i in range(max_length):
        chunk_values = []
        for means in agent_means.values():
            if i < len(means):
                chunk_values.append(means[i])
        overall_mean.append(np.mean(chunk_values))

    for agent, means in agent_means.items():
        plt.plot(means, label=agent)
    plt.plot(overall_mean, label="Overall", linestyle="--", linewidth=3)
    plt.xlabel(f"Step (x{step})")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.show()


file_path = "results/logs/app.log"

rewards = extract_rewards(file_path)
plot_rewards(rewards)

response_times = extract_response_times(file_path)
filtered_response_times = [
    response_time for response_time in response_times if response_time != 9999999.9
]
plot_response_times(filtered_response_times)
