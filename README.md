# A Performance-Aware and Adaptive Scheduling Mechanism for Microservices Placement in the Cloud-Fog Computing Continuum

---

With the advancement of technology and the emergence of diverse computing environments such as fog, edge, and cloud, the need for efficient mechanisms for scheduling and placing microservices has significantly increased.

This project designs and develops a novel mechanism for placing microservices in distributed and heterogeneous environments.

Leveraging a combination of reinforcement learning techniques and graph analysis, the proposed method aims to:

- **Reduce response time**
- **Optimize resource utilization**
- **Enhance system flexibility**

## System Overview

The system operates in three main phases:

1. **Dependency Graph Construction**  
   Build a graph based on microservice interactions and environment details.

2. **Initial Scheduling**  
   Partition the graph (using the Louvain algorithm) and place microservices considering resource and latency constraints.

3. **Optimization with Reinforcement Learning**  
   A multi-agent RL environment optimizes placements and migrations dynamically based on rewards.

Microservice migrations are triggered to improve response times, resource efficiency, and adapt to environmental or workload changes.

Simulation results validate the effectiveness of the approach, demonstrating improvements in system performance and adaptability.

---

## How to Run the Project

### Prerequisites

- Python 3.9 or later

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

The project is executed via the `main.py` script and supports two modes:

- **Without a setup file**: Generates a new microservices and environment setup.
- **With a setup file**: Loads an existing setup for training and evaluation.

#### Example Commands

**Run without a setup file:**

```bash
python main.py
```

**Run with a setup file:**

```bash
python main.py --setup_path path/to/setup_file.pkl
```

### What Happens During Execution

#### If a setup file is provided

- Loads microservices and network topology.
- Visualizes graphs.
- Trains the RL model based on the loaded configuration.

#### If no setup file is provided

- Generates a random microservices graph and network topology.
- Partitions the graph using the Louvain algorithm.
- Performs initial placement based on network and resource conditions.
- Creates a simulated workload population.
- Saves the generated setup for future reuse.
- Visualizes the graphs.
- Trains the RL model.

---

## Logs and Results

- Logs are stored at:  
  `results/logs/app.log` (default path)

- The best model checkpoint is saved after training, and its path is logged.

---

## Visualization

The system automatically visualizes:

- **Microservices Dependency Graph**
- **Network Topology Graph**

These visualizations help to better understand how microservices are grouped, scheduled, and migrated during optimization.
