# DQN Crasher

`dqn_crasher` uses RL controlled NPC agents to cause collisions in the [highway-env](https://highway-env.farama.org) gymnaisum environment.

## Installation

We recommend using [Conda](https://docs.conda.io) or [Mamba](https://mamba.readthedocs.io) for environment management.

```bash
git clone https://github.com/amarkoolk/dqn_crasher.git
cd dqn_crasher
pip install -e .
```


This will install all dependencies and make the `dqn_crasher` package (and its console scripts) available.

### Using `uv` (Recommended)

To set up the environment with `uv`:

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```
    This successfully installs the project with the specific `HighwayEnv` branch (`multi-car-scenario`).

3.  **Run commands**:
    prefix with `uv run`, e.g., `uv run dqn-crasher`.


## Usage

### Running Experiments

You can run the main experiment either as a Python module or via the installed console script:

```bash
# Using the Python module interface
python -m dqn_crasher.main --config configs/default.yaml

# Or using the console script
dqn-crasher --config configs/default.yaml
```

### Configuration

All experiments are configured via YAML files in the `src/dqn_crasher/configs/` directory. Example:

```yaml
env: "crash-v0"
seed: 42
episodes: 1000
dqn:
  learning_rate: 0.0005
  batch_size: 64
  epsilon_decay: 0.995
  ...
```

You can override config values via command-line flags:

```bash
python -m dqn_crasher.main --config configs/default.yaml dqn.learning_rate=0.001
```

### WandB Sweeps

To launch a hyperparameter sweep with [Weights & Biases](https://wandb.ai):

```bash
wandb sweep configs/sweep.yaml
wandb agent <SWEEP_ID>
```

See the `configs/` folder for ready-to-use sweep configs.

### Examples

Train a DQN agent and analyze crash scenarios:

```bash
python -m dqn_crasher.main
```

## Project Structure

```
dqn_crasher/
├── src/dqn_crasher/       # core package
│   ├── agents/            # DQN agent
│   ├── buffers/           # replay buffer, sum-tree
│   ├── configs/           # env + model configs (packaged)
│   ├── scenarios/         # scenarios & policies
│   ├── training/          # runner
│   ├── utils/             # helpers, logging
│   └── visualization/     # plotting
├── sweep_configs/         # WandB sweep configs (not packaged)
├── examples/              # experiment & plotting scripts
├── slurm_scripts/         # Slurm job scripts
├── models/                # trained models (git-ignored)
├── environment.yml        # Conda/Mamba env
└── pyproject.toml         # pip install entry
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact the maintainer at amar@example.com.
