"""
Simple SB3 DQN training script with SAT-based reward.

Wraps the multi-agent crash-v0 env into a single-agent env where:
  - The agent controls the NPC (adversarial)
  - The ego drives with a fixed policy (IDM / no-op)
  - Reward comes from the SAT reward wrapper
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import highway_env

from dqn_crasher.utils.config import load_pkg_yaml
from sat_reward_wrapper import SATRewardWrapper, VALID_CRASH_TYPES, R_MATCH, W_SHAPING

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class SingleAgentNPCWrapper(gym.Wrapper):
    """
    Unwraps the single-controlled-vehicle crash-v0 env from tuple spaces
    to standard Box/Discrete spaces for SB3 compatibility.

    With controlled_vehicles=1 and use_mobil=True, the env returns:
      obs: tuple of length 1 (just the NPC obs)
      action: tuple of length 1 (just the NPC action)
    This wrapper flattens that to a plain 1D obs and scalar action.
    The ego is driven by MOBIL/IDM internally by highway-env.
    """

    def __init__(self, env, frame_stack=5, n_features=5):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.n_features = n_features

        # Unwrap the single-element tuple obs space → flat 1D
        obs_space = env.observation_space[0]
        flat_dim = int(np.prod(obs_space.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

        # Unwrap the single-element tuple action space
        self.action_space = env.action_space[0]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs[0]), info

    def step(self, action):
        # Wrap scalar action back into tuple for the env
        obs, reward, terminated, truncated, info = self.env.step((int(action),))
        return self._flatten_obs(obs[0]), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        """Flatten observation to 1D vector with interleaved frame stacking."""
        n_cars = obs.shape[0]

        if self.frame_stack > 1:
            reshaped = obs.reshape((n_cars, self.n_features, self.frame_stack))
            interleaved = np.empty((n_cars * self.frame_stack, self.n_features), dtype=obs.dtype)
            for i in range(n_cars):
                interleaved[i::n_cars] = reshaped[i]
            return interleaved.flatten()
        else:
            return obs.flatten()


class CrashLoggingCallback(BaseCallback):
    """Logs crash rate, crash types, and episode rewards over training."""

    def __init__(self, target_crash_type, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.target_crash_type = target_crash_type
        self.log_interval = log_interval
        self.crashes = []
        self.crash_types = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_crash_types = []
        # Track per-step for current episode
        self._current_ep_reward = 0.0
        self._current_ep_len = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        self._current_ep_reward += float(rewards[0])
        self._current_ep_len += 1

        for info in infos:
            if dones[0]:
                crashed = info.get("crashed", False)
                self.crashes.append(int(crashed))
                ct = info.get("sat_crash_type", "no_crash") if crashed else "no_crash"
                self.episode_rewards.append(self._current_ep_reward)
                self.episode_lengths.append(self._current_ep_len)
                self.episode_crash_types.append(ct)
                if crashed:
                    self.crash_types.append(ct)
                self._current_ep_reward = 0.0
                self._current_ep_len = 0

        if self.num_timesteps % self.log_interval == 0 and len(self.crashes) > 0:
            window = min(100, len(self.crashes))
            recent_crashes = self.crashes[-window:]
            crash_rate = sum(recent_crashes) / len(recent_crashes)

            recent_types = self.episode_crash_types[-window:]
            target_hits = sum(1 for t in recent_types if t == self.target_crash_type)
            target_rate = target_hits / len(recent_types)
            wrong_crashes = sum(1 for t in recent_types if t not in ("no_crash", self.target_crash_type))
            wrong_rate = wrong_crashes / len(recent_types)

            print(
                f"  Step {self.num_timesteps:>7}: "
                f"crash={crash_rate:.1%}  "
                f"target({self.target_crash_type})={target_rate:.1%}  "
                f"wrong_type={wrong_rate:.1%}  "
                f"(last {window} eps)"
            )

        return True

    def print_summary(self):
        if not self.crash_types:
            print("  No crashes recorded.")
            return
        from collections import Counter
        counts = Counter(self.crash_types)
        total = len(self.crash_types)
        target_n = counts.get(self.target_crash_type, 0)
        print(f"\n{'='*50}")
        print(f"  Crash type distribution ({total} total crashes):")
        for ct, n in sorted(counts.items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if ct == self.target_crash_type else ""
            print(f"    {ct:25s}: {n:5d} ({n/total:.1%}){marker}")
        print(f"  Target hit rate: {target_n}/{total} = {target_n/total:.1%}")
        print(f"{'='*50}\n")


def make_env(target_crash_type, frame_stack=5, spawn_subset=None):
    gym_config = load_pkg_yaml("configs/env/crash_type_reward.yaml")
    gym_config["observation"]["observation_config"]["frame_stack"] = frame_stack
    if spawn_subset is not None:
        yaml_configs = set(gym_config["spawn_configs"])
        unknown = [c for c in spawn_subset if c not in yaml_configs]
        if unknown:
            raise ValueError(
                f"--spawn_subset contains configs not in the YAML pool: {unknown}. "
                f"Valid: {sorted(yaml_configs)}"
            )
        gym_config["spawn_configs"] = list(spawn_subset)

    env = gym.make("crash-v0", config=gym_config)
    env = SATRewardWrapper(env, target_crash_type=target_crash_type)
    env = SingleAgentNPCWrapper(env, frame_stack=frame_stack)
    env = Monitor(env)
    return env


def plot_training(callback, results_dir, target_crash_type, total_timesteps):
    episodes = np.arange(len(callback.episode_rewards))
    rewards = np.array(callback.episode_rewards)
    lengths = np.array(callback.episode_lengths)
    crashes = np.array(callback.crashes)
    window = 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"DQN — target: {target_crash_type} ({total_timesteps//1000}k steps)")

    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.15, color='blue', linewidth=0.5)
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(rewards)), smoothed, color='blue', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")

    ax = axes[1]
    ax.plot(episodes, lengths, alpha=0.15, color='green', linewidth=0.5)
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(lengths)), smoothed, color='green', linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")

    ax = axes[2]
    if len(crashes) > window:
        rolling_rate = np.convolve(crashes, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(crashes)), rolling_rate, color='red', linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"Crash Rate (rolling {window})")
    ax.set_xlabel("Episode")

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"training_curves_{target_crash_type}_{total_timesteps//1000}k.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plots saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="rear-end",
                        choices=sorted(VALID_CRASH_TYPES),
                        help="Crash type to reward (all others get 0)")
    parser.add_argument("--steps", type=int, default=750_000)
    parser.add_argument("--warm_start", default=None,
                        help="Path to a .zip checkpoint to continue training from")
    args = parser.parse_args()

    frame_stack = 5
    total_timesteps = args.steps
    target_crash_type = args.target
    results_dir = os.path.join(os.path.dirname(__file__), "..", "new_results", "sb3")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"SB3 DQN Training — target: {target_crash_type}")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Reward: +{R_MATCH} (target crash) / -R_WRONG (wrong type) + MTV shaping w={W_SHAPING}")
    if args.warm_start:
        print(f"Warm start: {args.warm_start}")
    print()

    env = make_env(target_crash_type=target_crash_type, frame_stack=frame_stack)

    callback = CrashLoggingCallback(target_crash_type=target_crash_type, log_interval=10000)

    if args.warm_start:
        model = DQN.load(args.warm_start, env=env)
        # Keep exploration low — model already knows the geometry
        model.exploration_initial_eps = 0.1
        model.exploration_final_eps   = 0.05
        model.exploration_fraction    = 0.2
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=128,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            train_freq=4,
            verbose=0,
            tensorboard_log=os.path.join(results_dir, "tb_logs"),
        )

    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=args.warm_start is None,
    )

    model_path = os.path.join(results_dir, f"dqn_{target_crash_type}_{total_timesteps//1000}k")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    callback.print_summary()
    plot_training(callback, results_dir, target_crash_type, total_timesteps)

    env.close()


if __name__ == "__main__":
    main()
