import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import gymnasium as gym
import highway_env
import torch
from dqn_crasher.utils.config import load_pkg_yaml
from dqn_crasher.utils.helpers import make_players
from dqn_crasher.utils.wandb_logging import initialize_logging

import dqn_crasher.scenarios.policies as policies
import dqn_crasher.scenarios.scenarios as scenarios
import wandb
from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.training.runner import MultiAgentRunner
from dqn_crasher.utils.utils import DeviceHelper
from dqn_crasher.scenarios.amago_policy import AMAGOPolicy



def build_new_dqn_policy(config, device, cycle, init_model, model_str, train : bool = False):
    trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
        f"{model_str}_{cycle}"
    )

    config["gym_config"] = load_pkg_yaml("configs/env/multi_agent_npc.yaml")

    tmp = gym.make(config["env_name"], config=config["gym_config"])
    act_space = tmp.action_space[0]
    n_act = act_space.n
    n_obs = 10 * config.get("frame_stack", 1)
    tmp.close()

    dqn_agent = DQN_Agent(n_obs, n_act, act_space, config, device)
    dqn_policy = policies.DQNPolicy(dqn_agent, trajectory_save_path, train = train, init_model = init_model)

    return dqn_agent, dqn_policy

def build_new_amago_policy(config, device):
    trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
        f"amago"
    )
    
    amago_policy = AMAGOPolicy(trajectory_store_dir=trajectory_save_path)
    amago_policy.policy.eval()

    return amago_policy

def run_policy_evaluation(config, device, policy_A, policy_B, model_str):

    gym_config = load_pkg_yaml("configs/env/multi_agent_npc.yaml")

    gym_config["observation"]["observation_config"]["frame_stack"] = config.get("frame_stack", 1)
    config["gym_config"] = gym_config
    run = initialize_logging(config, False, eval=True, run_name = f"{model_str}_eval_{config.get("model_pool")}")
    runner = MultiAgentRunner(config, device, policy_A, policy_B)
    runner.test()

    runner.env.close()
        


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("env_config")
    args = parser.parse_args()

    config = load_pkg_yaml(f"configs/model/{args.env_config}.yaml")
    device = DeviceHelper.get(config)

    model_a = '/p/crash/testest/models/EGO_1/model_98014.pth'
    model_b = '/p/crash/testest/models/NPC_1/model_100000.pth'

    policy_a = build_new_amago_policy(config, device)
    agent_b, policy_b = build_new_dqn_policy(config, device, 1, model_a, "EGO", False)
    # agent_b, policy_b = build_new_dqn_policy(config, device, 1, model_b, "NPC", False)

    policy_a.set_car_id(0)

    run_policy_evaluation(config, device, policy_a, policy_b, "EGO")


        


if __name__ == "__main__":
    main()
