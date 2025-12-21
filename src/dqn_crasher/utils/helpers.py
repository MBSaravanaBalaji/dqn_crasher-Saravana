import copy
import importlib
import os
from collections import deque
from typing import Callable, List, Sequence, Tuple, Union

import gymnasium as gym
import highway_env
import numpy as np
import torch

from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.scenarios import policies, scenarios
from dqn_crasher.scenarios.policies import (
    BasePolicy,
    DQNPolicy,
    MobilPolicy,
    PolicyDistribution,
    ScenarioPolicy,
)


def make_players(config, device):

    # Policy A
    p_A_list = config.get("policy_A")
    model_A_list = config.get("initial_models_A")
    class_list_A = []
    if len(p_A_list) == 1:
        p_A = pick_policy_function(p_A_list[0], config, device)
    elif len(p_A_list) > 1:
        for class_str in p_A_list:
            class_list_A.append(pick_policy_function(class_str, config, device))
        p_A = PolicyDistribution(class_list_A)
    else:
        raise Exception("Empty Policy List")

    # Policy B
    p_B_list = config.get("policy_B")
    model_B_list = config.get("initial_models_B")
    class_list_B = []
    if len(p_B_list) == 1:
        p_B = pick_policy_function(p_B_list[0], config, device)
    elif len(p_B_list) > 1:
        for class_str in p_B_list:
            config["initial_model"] = config.get("initial_model_B", None)
            class_list_B.append(pick_policy_function(class_str, config, device))
        p_B = PolicyDistribution(class_list_B)
    else:
        raise Exception("Empty Policy List")

    return p_A, p_B


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def initialize_stats(queue_len=100) -> dict:
    return {
        "num_crashes": {},
        "aggregate": {},
        "total_crashes": [],
        "episode_rewards": 0,
        "episode_duration": 0,
        "ego_speed": 0,
        "npc_speed": 0,
        "ep_rew_total": deque([], maxlen=queue_len),
        "ep_len_total": deque([], maxlen=queue_len),
        "ego_speed_total": deque([], maxlen=queue_len),
        "npc_speed_total": deque([], maxlen=queue_len),
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.0,
        "collision_reward": 0.0,
        "ttc_x_reward": 0.0,
        "ttc_y_reward": 0.0,
        "epsilon": 0.0,
        "spawn_config": None,
        "episode_num": 0,
        "policy_name": None,
        "specific_policy_name": None,
        "policy_type": None,
        "training_step": 0,
        "testing_step": 0,
        "checkpoint_step": 0,
        "metrics_type": "training",
        "checkpoint_testing": False,
        "crashes_by_type": {},
        "crashes_by_contact": {},
        "crashes_by_ego_feature": {},
        "crashes_by_npc_feature": {},
        # Cumulative counters that persist across episodes
        "total_crashes_by_type": {},
    }


def populate_stats(info, episode_statistics: dict):
    episode_statistics["scenario"] = info["scenario"]
    episode_statistics["num_retries"] = info["num_retries"]
    episode_statistics["episode_duration"] += 1
    episode_statistics["ego_speed"] += info["ego_speed"]
    episode_statistics["npc_speed"] += info["npc_speed"]
    episode_statistics["right_lane_reward"] += info["rewards"]["right_lane_reward"]
    episode_statistics["high_speed_reward"] += info["rewards"]["high_speed_reward"]
    episode_statistics["ttc_x_reward"] += info["rewards"]["ttc_x_reward"]
    episode_statistics["ttc_y_reward"] += info["rewards"]["ttc_y_reward"]
    episode_statistics["collision_reward"] += info["rewards"]["collision_reward"]
    episode_statistics["episode_rewards"] += info["rewards"]["total"]
    episode_statistics["epsilon"] = info["eps_threshold"]

    if "collision_details" in info:
        cd = info["collision_details"]
        for key, stat_map in [
            ("collision_type", "crashes_by_type"),
            ("contact_type", "crashes_by_contact"),
            ("ego_feature", "crashes_by_ego_feature"),
            ("npc_feature", "crashes_by_npc_feature"),
        ]:
            val = cd.get(key, "unknown")
            if val not in episode_statistics[stat_map]:
                episode_statistics[stat_map][val] = 0
            episode_statistics[stat_map][val] += 1
            
            # Update cumulative counters for types (persists)
            if key == "collision_type":
                if val not in episode_statistics["total_crashes_by_type"]:
                    episode_statistics["total_crashes_by_type"][val] = 0
                episode_statistics["total_crashes_by_type"][val] += 1


def reset_stats(stats: dict, preserve_episode_num=False):
    # Save counters and flags if we need to preserve them
    counters_to_preserve = {}
    if preserve_episode_num:
        # Save all step counters and metadata
        counters_to_preserve = {
            "episode_num": stats.get("episode_num", 0),
            "training_step": stats.get("training_step", 0),
            "testing_step": stats.get("testing_step", 0),
            "checkpoint_step": stats.get("checkpoint_step", 0),
            "metrics_type": stats.get("metrics_type", "training"),
            "checkpoint_testing": stats.get("checkpoint_testing", False),
            "policy_name": stats.get("policy_name", None),
            "specific_policy_name": stats.get("specific_policy_name", None),
            "policy_type": stats.get("policy_type", None),
            "scenario": stats.get("scenario", None),
            "spawn_config": stats.get("spawn_config", None),
        }

    stats["episode_rewards"] = 0
    stats["episode_duration"] = 0
    stats["ego_speed"] = 0
    stats["npc_speed"] = 0
    stats["right_lane_reward"] = 0.0
    stats["high_speed_reward"] = 0.0
    stats["collision_reward"] = 0.0
    stats["ttc_x_reward"] = 0.0
    stats["ttc_y_reward"] = 0.0

    stats["crashes_by_type"] = {}
    stats["crashes_by_contact"] = {}
    stats["crashes_by_ego_feature"] = {}
    stats["crashes_by_npc_feature"] = {}

    # Restore all preserved counters or increment episode_num
    if preserve_episode_num and counters_to_preserve:
        # Restore all counters and metadata
        for key, value in counters_to_preserve.items():
            stats[key] = value
    else:
        # Just increment the episode number for normal progress
        stats["episode_num"] = stats.get("episode_num", 0) + 1


def interleave_stacked_frames(obs, n_cars, n_features, n_stack):
    if n_stack > 1:
        reshaped_obs = obs.reshape((obs.shape[0], n_features, n_stack))
        interleaved_obs = np.empty((n_cars * n_stack, n_features), dtype=obs.dtype)
        for i in range(n_cars):
            interleaved_obs[i::n_cars] = reshaped_obs[i]
    else:
        interleaved_obs = obs

    return interleaved_obs.flatten()


def obs_to_state(obs, n_observations, device, num_features=5, frame_stack=1):
    if len(obs) == 2:
        flattened_ego_obs = obs[0].flatten()
        ego_obs = interleave_stacked_frames(obs[0], len(obs), num_features, frame_stack)
        ego_state = torch.tensor(
            ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
        )
        npc_obs = interleave_stacked_frames(obs[1], len(obs), num_features, frame_stack)
        npc_state = torch.tensor(
            npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
        )
    elif len(obs) == 1:
        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[:n_observations]
        ego_state = torch.tensor(
            ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
        )
        npc_state = torch.tensor(
            ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
        )
        npc_state[0, 0] = 1.0
        npc_state[0, 1] = ego_state[0, 6]
        npc_state[0, 2] = ego_state[0, 7]
        npc_state[0, 3] = ego_state[0, 8]
        npc_state[0, 4] = ego_state[0, 9]
        npc_state[0, 5] = ego_state[0, 0]
        npc_state[0, 6] = ego_state[0, 1]
        npc_state[0, 7] = ego_state[0, 2]
        npc_state[0, 8] = ego_state[0, 3]
        npc_state[0, 9] = ego_state[0, 4]

    return ego_state, npc_state


def make_step_actions(ego_action, npc_action, vs_mobil=False):
    # if we’re “versus MOBIL” then MOBIL is our ego,
    # so we send (npc, npc)
    return (npc_action, npc_action) if vs_mobil else (ego_action, npc_action)


def pick_policy_function(class_str: str, config, device):
    strings = class_str.split(".")
    if "scenario" in class_str:
        class_type = class_from_path(class_str)
        return set_scenario_policy(class_type, config)
    elif "Mobil" in class_str and len(strings) > 2:
        class_type = class_from_path(strings[0] + "." + strings[1])
        config["spawn_config"] = [strings[2]]
        return set_policy(class_type, config, device)
    else:
        class_type = class_from_path(class_str)
        return set_policy(class_type, config, device)


def set_policy(class_type, config, device):
    trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
    )

    if class_type == DQNPolicy:
        gym_config = config["gym_config"]
        tmp = gym.make(config["env_name"], config=gym_config)
        act_space = tmp.action_space[0]
        n_act = act_space.n
        n_obs = 10 * config.get("frame_stack", 1)
        tmp.close()

        agent = DQN_Agent(n_obs, n_act, act_space, config, device)
        if config.get("train_ego", False) is False:
            agent.load_model(config.get("ego_model", ""))
        dqn_policy = class_type(
            agent,
            trajectory_save_path,
            config.get("train_ego", False),
            config.get("initial_model", None),
        )
        return dqn_policy
    elif class_type == MobilPolicy:
        mobil_policy = MobilPolicy(trajectory_save_path, config["spawn_config"])
        return mobil_policy


def set_scenario_policy(scenario_type, config):
    n_obs = 10 * config.get("frame_stack", 1)
    if scenario_type == scenarios.IdleFaster:
        scen = ScenarioPolicy(scenarios.IdleFaster, n_obs, config)
    elif scenario_type == scenarios.IdleSlower:
        scen = ScenarioPolicy(scenarios.IdleSlower, n_obs, config)
    elif scenario_type == scenarios.CutIn:
        scen = ScenarioPolicy(scenarios.CutIn, n_obs, config)
    elif scenario_type == scenarios.CutInSlowDown:
        scen = ScenarioPolicy(scenarios.CutInSlowDown, n_obs, config)
    else:
        raise Exception("No Viable Scenario Chosen")

    return scen
