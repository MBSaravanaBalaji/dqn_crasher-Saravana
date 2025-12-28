# runner.py
import copy
import os
from logging import raiseExceptions

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
from tqdm import tqdm

import dqn_crasher.scenarios.policies as policies
from dqn_crasher.scenarios.amago_policy import AMAGOPolicy
import dqn_crasher.utils.helpers as helpers
import wandb
from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.buffers.experience_replay import Transition
from dqn_crasher.utils.wandb_logging import (
    initialize_logging,
    log_checkpoint_summary,
    log_stats,
)


class MultiAgentRunner:
    def __init__(self, config, device, policy_a, policy_b, update_elo: bool = False):
        self.env_name = config["env_name"]
        self.cfg: dict = config
        self.gym_cfg: dict = config["gym_config"]
        self.dev = device
        self.A : policies.BasePolicy = policy_a
        self.B : policies.BasePolicy = policy_b
        self.update_elo = update_elo

        # Create global dictionary to track specific policy stats
        if "specific_policy_stats" not in globals():
            globals()["specific_policy_stats"] = {}

        # a single env instance just to pull action_space / obs dims
        tmp = gym.make(self.env_name, config=self.gym_cfg)
        self.action_space = tmp.action_space[0]
        self.n_actions = self.action_space.n
        self.n_obs = 10 * config.get("frame_stack", 1)
        tmp.close()

    def save_model(self, train_player, step_number: int = 0):
        if train_player == "A" and type(self.A.agent) == DQN_Agent:
            self.A.save_model(
                os.path.join(
                    self.cfg.get("root_directory", ""),
                    self.cfg.get("model_save_path", "models/model")
                    + f"/model_{step_number}.pth",
                )
            )
        elif type(self.B.agent) == DQN_Agent:
            self.B.save_model(
                os.path.join(
                    self.cfg.get("root_directory", ""),
                    self.cfg.get("model_save_path", "models/model")
                    + f"/model_{step_number}.pth",
                )
            )

    def train(self, train_player="A"):
        total_timesteps = self.cfg.get("total_timesteps", 100000)
        test_interval = self.cfg.get(
            "test_interval", 10000
        )  # Run tests every 10k steps by default
        t = 0
        pbar = tqdm(total=total_timesteps)

        if self.cfg.get("track", False):
            stats = helpers.initialize_stats()

        if self.cfg.get("save_trajectories", False):
            self.A.store.save_metadata(self.cfg)
            self.A.test_store.save_metadata(self.cfg)

        while t < total_timesteps:
            self.A.reset()
            self.B.reset()
            self._set_config()
            # Use t as the training step counter
            stats["training_step"] = t
            steps = self._run_episode(
                train_player=train_player,
                t_start=t,
                stats=stats,
                pbar=pbar,
                checkpoint_step=None,
            )
            t += steps

            # Run incremental testing at specified intervals
            if (
                self.cfg.get("enable_checkpoint_testing", True)
                and t > 0
                and t % test_interval < steps
            ):
                # Use the current step as the checkpoint identifier
                self.test_checkpoint(checkpoint_step=t)
                self.save_model(train_player, t)

        self.save_model(train_player, t)

        if pbar:
            pbar.close()
        wandb.finish() 

    def test(self):
        total_eps = self.cfg.get("testing_episodes", 10)
        eps_done = 0

        if self.cfg.get("save_trajectories", False):
            self.A.store.save_metadata(self.cfg)
            self.A.test_store.save_metadata(self.cfg)

        self.test_checkpoint(0)

        wandb.finish()

    def test_checkpoint(self, checkpoint_step):
        """Run a smaller test during training to track progress"""
        total_eps = self.cfg.get(
            "checkpoint_testing_episodes", 5
        )  # Fewer episodes for checkpoints

        # Save current state to restore after testing
        save_state = (self.A.save_state(), self.B.save_state())

        # Track is already True if we're in training
        stats = helpers.initialize_stats()
        # Mark these stats as belonging to checkpoint testing
        stats["checkpoint_testing"] = True
        stats["metrics_type"] = "checkpoint_episode"

        scenario_vs_mobil = False
        if isinstance(self.A, policies.PolicyDistribution):
            scenario_vs_mobil = isinstance(self.A.policies[0], policies.ScenarioPolicy)

        if isinstance(self.A, policies.DQNPolicy):
            self.A.set_train(False)
            self.A.test_store.reset_filepath(checkpoint_step)

        if isinstance(self.B, policies.DQNPolicy):
            self.B.set_train(False)
            self.B.test_store.reset_filepath(checkpoint_step)

        if isinstance(self.B, policies.PolicyDistribution):
            for policy in self.B.policies:
                policy.test_store.reset_filepath(checkpoint_step)

        if scenario_vs_mobil:
            policy_iterate = self.A.policies
        else:
            if isinstance(self.B, policies.PolicyDistribution):
                policy_iterate = self.B.policies
            else:
                policy_iterate = [self.B]

        for i in tqdm(range(len(policy_iterate))):
            if isinstance(self.A, AMAGOPolicy):
                self.A.reset(test=True)
            if isinstance(self.B, AMAGOPolicy):
                self.B.reset(test=True)
            for j in tqdm(range(total_eps), leave = False):
                if scenario_vs_mobil:
                    self.A.reset(test=True)
                    self.B.reset(test=True)
                else:
                    if not isinstance(self.A, AMAGOPolicy):
                        self.A.reset(test=True)
                    if not isinstance(self.B, AMAGOPolicy):
                        self.B.reset(test=True)
                self._set_config()

                # Run the episode
                stats["metrics_type"] = "checkpoint_episode"
                episode_results = self._run_episode(
                    train_player=None,
                    t_start=0,
                    stats=stats,
                    pbar=None,
                    checkpoint_step=checkpoint_step,
                )

        if self.cfg.get("track", False):
            log_checkpoint_summary(stats, checkpoint_step)

        self.A.restore_state(save_state[0])
        self.B.restore_state(save_state[1])

    def _set_config(self):
        def unwrap(policy_obj):
            # if it's a distribution, grab the inner scenario policy
            return (
                policy_obj.current_policy
                if isinstance(policy_obj, policies.PolicyDistribution)
                else policy_obj
            )

        scenario_policy = False
        mobil_policy = False
        for name in ("A", "B"):
            policy = unwrap(getattr(self, name))
            # ScenarioPolicy always gets configured
            if isinstance(policy, policies.ScenarioPolicy):
                policy.set_config(self.cfg)
                scenario_policy = True
            # If there is no Scenario Policy use other policy
            elif isinstance(policy, policies.MobilPolicy):
                mobil_policy = True
                if not scenario_policy:
                    policy.set_config(self.cfg)

        if mobil_policy is False:
            self.cfg["gym_config"]["vs_mobil"] = False
            self.cfg["gym_config"]["use_mobil"] = False
            self.cfg["gym_config"]["controlled_vehicles"] = 2
            self.cfg["gym_config"]["other_vehicles"] = 0
        else:
            self.cfg["gym_config"]["vs_mobil"] = True
            self.cfg["gym_config"]["use_mobil"] = True
            self.cfg["gym_config"]["controlled_vehicles"] = 1
            self.cfg["gym_config"]["other_vehicles"] = 1

        self.gym_cfg = self.cfg["gym_config"]

    def _run_episode(self, train_player, t_start, stats, pbar, checkpoint_step=None):
        total_timesteps = self.cfg.get("total_timesteps", 100000)

        train_A = False
        train_B = False
        if train_player == "A":
            train_A = True
        elif train_player == "B":
            train_B = True

        env = gym.make(self.env_name, config=self.gym_cfg, render_mode="rgb_array")

        # Record videos when crash
        video_folder = os.path.join(self.cfg.get("root_directory", "results"), "videos")
        # Use t_start to give a unique name, preventing overwrite since we recreate env each time
        video_prefix = f"crash_step_{t_start}"
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True, # Record candidate, will delete if no crash
            name_prefix=video_prefix,
            disable_logger=True
        )
        obs, info = env.reset()
        reward = None
        term = None
        trunc = None
        ego_s, npc_s = helpers.obs_to_state(
            obs, self.n_obs, self.dev, frame_stack=self.cfg["frame_stack"]
        )
        
        self.A.set_state(ego_s, npc_s)
        self.B.set_state(npc_s, ego_s)

        # Classify type of scenario
        classification_source = None
        scenario = None
        if train_player:
            if isinstance(self.B, policies.PolicyDistribution):
                classification_source = self.B
        elif isinstance(self.A, policies.PolicyDistribution):
            if isinstance(self.A.current_policy, policies.ScenarioPolicy):
                classification_source = self.A
        else:
            classification_source = self.B

        if isinstance(classification_source, policies.PolicyDistribution):
            if isinstance(
                classification_source.current_policy, policies.ScenarioPolicy
            ):
                scenario = str(
                    type(classification_source.current_policy.scenario).__name__
                )
            elif isinstance(classification_source.current_policy, policies.MobilPolicy):
                scenario = (
                    str(type(classification_source.current_policy).__name__)
                    + "."
                    + classification_source.current_policy.spawn_configs[0]
                )
            elif isinstance(classification_source.current_policy, policies.DQNPolicy):
                scenario = (
                    str(type(classification_source.current_policy).__name__)
                    + str(classification_source.policy_idx)
                )


        if scenario is None and "spawn_config" in info.keys():
            scenario = info["spawn_config"]

        if self.update_elo:
            if self.B.adversarial:
                model_str = "NPC"
            else:
                model_str = "EGO"
            scenario = f"{model_str}{self.B.policy_idx + 1}"

        if self.cfg.get("save_trajectories", False):
            if train_player:
                self.A.store.start_episode(stats["episode_num"])
                self.B.store.start_episode(stats["episode_num"])
            else:
                self.A.test_store.start_episode(stats["episode_num"])
                self.B.test_store.start_episode(stats["episode_num"])

        t = t_start
        done = False
        steps = 0

        while not done and t < total_timesteps:
            if isinstance(self.A, AMAGOPolicy):
                a_A, action_logits_A = self.A.select_action(obs, reward, term, trunc, info)
            else:
                action_logits_A = self.A.select_action(ego_s, npc_s, t)
                a_A = torch.argmax(action_logits_A)

            if isinstance(self.B, AMAGOPolicy):
                a_B, action_logits_B = self.B.select_action(obs, reward, term, trunc, info)
            else:
                action_logits_B = self.B.select_action(npc_s, ego_s, t)
                a_B = torch.argmax(action_logits_B)

            actions = (a_A.cpu().numpy(), a_B.cpu().numpy())

            # env step
            obs, reward, term, trunc, info = env.step(actions)
            done = term or trunc

            # Extract collision classification if crashed
            if info.get("crashed", False):
                try:
                    # Access the base environment to get the road objects
                    base_env = env.unwrapped
                    if hasattr(base_env, "road"):
                        for v in base_env.road.vehicles:
                            # look for the ego vehicle or any vehicle that has collision data
                            if hasattr(v, "collision_classification") and v.collision_classification:
                                cc = v.collision_classification
                                
                                # side-swipe direction based on ego pov
                                c_type = cc.collision_type
                                if c_type == "side-swipe":
                                    if "left" in cc.ego_feature:
                                        c_type = "side-swipe-left"
                                    elif "right" in cc.ego_feature:
                                        c_type = "side-swipe-right"

                                info["collision_details"] = {
                                    "collision_type": c_type,
                                    "contact_type": cc.contact_type,
                                    "ego_feature": cc.ego_feature,
                                    "npc_feature": cc.npc_feature,
                                }
                                break

                except Exception:
                    pass


            if self.cfg.get("render", False):
                env.render()

            next_A, next_B = helpers.obs_to_state(
                obs, self.n_obs, self.dev, frame_stack=self.cfg["frame_stack"]
            )
            transition_A = Transition(
                ego_s, a_A, next_A, torch.tensor(reward, device=self.dev)
            )
            transition_B = Transition(
                npc_s, a_B, next_B, torch.tensor(reward, device=self.dev)
            )

            store_transition_A = Transition(
                ego_s, action_logits_A, next_A, torch.tensor(reward, device=self.dev)
            )
            store_transition_B = Transition(
                npc_s, action_logits_B, next_A, torch.tensor(reward, device=self.dev)
            )

            ego_s = self.A.update(transition_A, term, train_A)
            npc_s = self.B.update(transition_B, term, train_B)

            self.A.set_state(ego_s, npc_s)
            self.B.set_state(npc_s, ego_s)

            # store trajectories for the  policy
            if self.cfg.get("save_trajectories"):
                if train_player:
                    self.A.store.add(store_transition_A, info)
                    self.B.store.add(store_transition_B, info)
                else:
                    self.A.test_store.add(store_transition_A, info)
                    self.B.test_store.add(store_transition_B, info)

            if train_player:
                info["eps_threshold"] = self.A.agent.eps_threshold
            else:
                info["eps_threshold"] = 0.0

            info["rewards"]["total"] = reward
            info["ego_speed"] = (
                ego_s[0, -7].cpu().numpy() ** 2 + ego_s[0, -6].cpu().numpy() ** 2
            ) ** 0.5
            info["npc_speed"] = (
                npc_s[0, -7].cpu().numpy() ** 2 + npc_s[0, -6].cpu().numpy() ** 2
            ) ** 0.5
            info["scenario"] = scenario


            # Populate the main stats dictionary
            helpers.populate_stats(info, episode_statistics=stats)

            t += 1
            steps += 1
            if pbar:
                pbar.update(1)

        if self.update_elo:
            ego = not self.B.adversarial
            self.B.update_elos(int(info["crashed"]))


        if self.cfg.get("track", False):
            # Determine which metric type we're using based on context flags
            metrics_type = stats.get("metrics_type", "training")

            if metrics_type == "checkpoint_episode":
                # For checkpoint testing episodes - use testing_step
                current_step = stats.get("testing_step", 0)
                log_stats(info, stats, checkpoint=True, checkpoint_step=current_step)

                # Reset stats but preserve all counters and metadata
                helpers.reset_stats(stats, preserve_episode_num=False)
            elif metrics_type == "checkpoint_summary":
                # For checkpoint summary metrics - use checkpoint_step
                current_step = stats.get("checkpoint_step", 0)
                log_stats(info, stats, checkpoint=True, checkpoint_step=current_step)
                helpers.reset_stats(stats, preserve_episode_num=True)
            elif metrics_type == "testing":
                # For evaluation testing - use normal episode numbers
                log_stats(info, stats, checkpoint=True)
                helpers.reset_stats(stats, preserve_episode_num=False)
            else:
                # Normal training logging - use training_step
                current_step = stats.get("training_step", 0)
                log_stats(info, stats)
                helpers.reset_stats(stats, preserve_episode_num=False)

        if self.cfg.get("save_trajectories", False):
            if train_player:
                self.A.store.end_episode()
                self.B.store.end_episode()
            else:
                self.A.test_store.end_episode()
                self.B.test_store.end_episode()

        env.close()

        # keep video if crash occured and renames based on type of crash
        video_path = os.path.join(video_folder, f"{video_prefix}-episode-0.mp4")
        if os.path.exists(video_path):
            if not info.get("crashed", False):
                try:
                    os.remove(video_path)
                except OSError:
                    pass # Best effort cleanup
            else:
                try:
                    c_type = info.get("collision_details", {}).get("collision_type", "unknown_crash")
                    new_name = f"{c_type}_step_{t_start}-episode-0.mp4"
                    new_path = os.path.join(video_folder, new_name)
                    os.rename(video_path, new_path)
                except OSError:
                    pass # Keep original name if rename fails

        return t
