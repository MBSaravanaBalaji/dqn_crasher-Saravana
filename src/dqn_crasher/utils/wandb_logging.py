import os

import numpy as np


import wandb
os.environ["WANDB_SILENT"] = "true"



def initialize_logging(
    config,
    train_ego=None,
    eval=False,
    checkpoint=False,
    npc_pool_size=None,
    ego_pool_size=None,
    run_name=None,
):
    if wandb.run is not None:
        wandb.finish()

    is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))

    gym_config = config.get("gym_config", {})

    run_config = {
        "learning_rate": config.get("learning_rate", 5e-4),
        "architecture": config.get("architecture", "DQN"),
        "dataset": "Highway-Env",
        "max_steps": config.get("total_timesteps", 100000),
        "collision_reward": gym_config.get("collision_reward", 400),
        "ttc_x_reward": gym_config.get("ttc_x_reward", 4),
        "ttc_y_reward": gym_config.get("ttc_y_reward", 1),
        "batch_size": config.get("batch_size", 32),
        "gamma": config.get("gamma", 0.8),
        "eps_start": config.get("start_e", 1.0),
        "eps_end": config.get("end_e", 0.05),
        "eps_decay": config.get("decay_e", 6000),
        "tau": config.get("tau", 0.005),
        "ReplayBuffer": config.get("buffer_type", "ER"),
        "eval": eval,
        "checkpoint_testing": config.get("enable_checkpoint_testing", True),
        "test_interval": config.get("test_interval", 10000),
    }

    run_config["ego_version"] = config.get("ego_version", 0)
    run_config["npc_version"] = config.get("npc_version", 0)
    if train_ego is not None:
        run_config["train_ego"] = train_ego
    if npc_pool_size is not None:
        run_config["npc_pool"] = True
        run_config["npc_pool_size"] = npc_pool_size
    else:
        run_config["npc_pool"] = False
    if ego_pool_size is not None:
        run_config["ego_pool"] = True
        run_config["ego_pool_size"] = ego_pool_size
    else:
        run_config["ego_pool"] = False

    if config.get("wandb_tag", None) is not None:
        tags = config.get("wandb_tag").split(",")
    else:
        tags = []

    init_args = {"config": run_config, "tags": tags}

    if run_name is not None:
        init_args["name"] = run_name
    else:
        if config.get("run_name", "") is not "":
            init_args["name"] = config.get("run_name")

    if not is_sweep:
        init_args["project"] = config.get("wandb_project_name", "safetyh")

    run = wandb.init(**init_args)

    if is_sweep:
        config.update(dict(wandb.config))

    # Define separate x-axes for different metric types
    # Training metrics use training_step
    wandb.define_metric("training_step")
    wandb.define_metric("training/*", step_metric="training_step")

    # Testing episode metrics use testing_step
    wandb.define_metric("testing_step")
    wandb.define_metric("checkpoint_episode/*", step_metric="testing_step")

    # Checkpoint summary metrics use checkpoint_step
    wandb.define_metric("checkpoint_step")
    wandb.define_metric("checkpoint_summary/*", step_metric="checkpoint_step")

    return run


def log_evaluation(args, n_cycle):
    if wandb.run is not None:
        wandb.finish()

    run_config = {"cycle": n_cycle}

    run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandb_project_name,
        # track hyperparameters and run metadata
        config=run_config,
    )
    return run


def log_stats(info, episode_statistics: dict, checkpoint=False, checkpoint_step=None):
    # Breakdown scenario stats
    scenario_stat_keys = [
        "num_crashes",
        "episode_rewards",
        "episode_duration",
        "ego_speed",
        "npc_speed",
        "ttc_x_reward",
        "ttc_y_reward",
        "right_lane_reward",
        "high_speed_reward",
        "collision_reward",
        "num_retries",
    ]
    for stat in scenario_stat_keys:
        if stat not in episode_statistics["aggregate"]:
            episode_statistics["aggregate"][stat] = {}
        if info["scenario"] not in episode_statistics["aggregate"][stat]:
            episode_statistics["aggregate"][stat][info["scenario"]] = []
        if stat == "num_crashes":
            episode_statistics["aggregate"][stat][info["scenario"]].append(
                info["crashed"]
            )
        elif stat == "num_retries":  # -------- NEW --------
            episode_statistics["aggregate"][stat][info["scenario"]].append(
                info["num_retries"]
            )
        else:
            episode_statistics["aggregate"][stat][info["scenario"]].append(
                episode_statistics[stat] / episode_statistics["episode_duration"]
            )

    episode_statistics["total_crashes"].append(info["crashed"])

    # Calculate episode statistics safely
    episode_duration = max(
        1, episode_statistics.get("episode_duration", 1)
    )  # Avoid div/0

    # Get statistics with defaults to prevent KeyError
    ep_rewards = episode_statistics.get("episode_rewards", 0)
    ep_duration = episode_statistics.get("episode_duration", 1)
    ego_speed = episode_statistics.get("ego_speed", 0)
    npc_speed = episode_statistics.get("npc_speed", 0)
    num_retries = episode_statistics.get("num_retries", 0)


    # Add statistics to deques
    episode_statistics["ep_rew_total"].append(ep_rewards)
    episode_statistics["ep_len_total"].append(ep_duration)
    episode_statistics["ego_speed_total"].append(ego_speed / episode_duration)
    episode_statistics["npc_speed_total"].append(npc_speed / episode_duration)

    # Use more descriptive prefixes for different metric types
    if checkpoint:
        prefix = "checkpoint"  # Individual checkpoint testing episodes
    else:
        prefix = "training"  # Regular training metrics

    # Determine which step metric to use
    if checkpoint:
        step_metric = "testing_step"
    else:
        step_metric = "training_step"

    # For crash rate calculation, use min to avoid division by zero
    total_crashes = episode_statistics.get("total_crashes", [])
    crash_rate_divisor = min(100, len(total_crashes))
    config_crash_rate_divisor = min(
        100, len(episode_statistics.get("num_crashes", {}).get(info["scenario"], []))
    )

    # Get all values with safe defaults
    ep_rew_total = episode_statistics.get("ep_rew_total", [0])
    ep_len_total = episode_statistics.get("ep_len_total", [0])
    ego_speed = episode_statistics.get("ego_speed", 0)
    npc_speed = episode_statistics.get("npc_speed", 0)
    epsilon = episode_statistics.get("epsilon", 0)
    collision_reward = episode_statistics.get("collision_reward", 0)
    episode_duration = max(1, episode_statistics.get("episode_duration", 1))

    # Build log dictionary with safe calculations
    wandb_log = {
        f"{prefix}/ep_rew_mean": sum(ep_rew_total) / max(1, len(ep_rew_total)),
        f"{prefix}/ep_len_mean": sum(ep_len_total) / max(1, len(ep_len_total)),
        f"{prefix}/num_crashes": sum(total_crashes),
        f"{prefix}/sr100": sum(total_crashes[-crash_rate_divisor:])
        / max(1, crash_rate_divisor),
        f"{prefix}/ego_speed_mean": ego_speed / episode_duration,
        f"{prefix}/npc_speed_mean": npc_speed / episode_duration,
        f"{prefix}/scenario": info["scenario"],
        f"{prefix}/epsilon": epsilon,
        f"{prefix}/num_retries": info.get("num_retries", 0),
        f"{prefix}/collision_reward": collision_reward / episode_duration,
        f"{prefix}/{info['scenario']}/ego_speed_mean": ego_speed / episode_duration,
        f"{prefix}/{info['scenario']}/npc_speed_mean": npc_speed / episode_duration,
        f"{prefix}/{info['scenario']}/crash": int(info["crashed"]),
        f"{prefix}/{info['scenario']}/num_retries": info.get("num_retries", 0),
    }

    # Only add these metrics if we have crash data for this spawn config
    num_crashes = episode_statistics.get("num_crashes", {})
    if info["scenario"] in num_crashes and len(num_crashes[info["scenario"]]) > 0:
        # Get safe crashes for this config and calculate metrics
        config_crashes = num_crashes[info["scenario"]]
        crash_count = sum(config_crashes)
        sr100 = sum(config_crashes[-config_crash_rate_divisor:]) / max(
            1, config_crash_rate_divisor
        )

        wandb_log.update(
            {
                f"{prefix}/{info['scenario']}/num_crashes": crash_count,
                f"{prefix}/{info['scenario']}/sr100": sr100,
                f"{prefix}/{info['scenario']}/collision_reward": collision_reward
                / episode_duration,
            }
        )

    # Log detailed collision stats
    # Filter for only specific keys as requested
    allowed_keys = ["rear-end", "side-swipe", "side-swipe-left", "side-swipe-right"]

    if "total_crashes_by_type" in episode_statistics:
        for key, count in episode_statistics["total_crashes_by_type"].items():
            # Check if the key contains any of our allowed types (e.g. "side-swipe" matches "side-swipe")
            if any(allowed in key for allowed in allowed_keys):
                wandb_log[f"{prefix}/crashes/collision_type/{key}"] = count

    # Get the current episode number or checkpoint step
    current_step = episode_statistics.get("episode_num", 0)

    # Add the step metric to the log dict
    if step_metric == "checkpoint_step":
        wandb_log["checkpoint_step"] = current_step
    elif step_metric == "testing_step":
        wandb_log["testing_step"] = current_step
    else:
        wandb_log["training_step"] = episode_statistics.get("training_step", 0)

    # Add identifiers to clarify metric types and grouping
    if checkpoint:
        wandb_log["checkpoint_id"] = checkpoint_step
        wandb_log["metric_collection"] = (
            "checkpoint_testing" if checkpoint else "checkpoint_summary"
        )
    else:
        wandb_log["metric_collection"] = "training"

    # Log with custom tags (step is automatically handled by define_metric)
    wandb.log(wandb_log)


def log_checkpoint_summary(stats, checkpoint_step):
    prefix = "checkpoint_summary"
    wandb_log = {"checkpoint_step": checkpoint_step}
    scenario_stat_keys = [
        "num_crashes",
        "episode_rewards",
        "ego_speed",
        "npc_speed",
        "ttc_x_reward",
        "ttc_y_reward",
        "right_lane_reward",
        "high_speed_reward",
        "collision_reward",
    ]
    for stat in scenario_stat_keys:
        total_stat = []
        for scenario in stats["aggregate"][stat]:
            avg_stat = np.mean(stats["aggregate"][stat][scenario])
            wandb_log[f"{prefix}/{scenario}/{stat}"] = avg_stat
            total_stat.append(avg_stat)
        wandb_log[f"{prefix}/{stat}"] = np.mean(total_stat)

    wandb.log(wandb_log)
