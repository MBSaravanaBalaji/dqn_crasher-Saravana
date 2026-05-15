"""
SB3 DQN training for master NPC: NPC learns to set up scenarios where
the ego (IDM/MOBIL) causes crashes of a specific type.

Rear-end strategy (cut-in from adjacent lane, no-MOBIL):
  NPC starts in adjacent lane ahead of ego (forward_left or forward_right). Ego is
  faster (closing speed ~12 m/s). NPC waits in adjacent lane while ego closes the gap,
  then cuts in when gap is in the crash window (~15-22m). During the lane change, ego's
  IDM detects NPC only after NPC is halfway into ego's lane — by then the gap is too
  small to stop at ACC_MAX → ego's front hits NPC's rear → "rear-end".
  Use --no_mobil so ego cannot escape via MOBIL lane-change.

Side-swipe strategy:
  NPC positions alongside ego in the adjacent lane and forces a lane-change
  conflict via MOBIL incentives.

Usage:
    # Rear-end cut-in curriculum (recommended defaults):
    uv run python crash_type_reward/master/train_sb3.py --target rear-end \
        --apply_rear_end_defaults --steps 2000000

    # Or specify knobs manually:
    uv run python crash_type_reward/master/train_sb3.py --target rear-end \
        --spawn_subset forward_left,forward_right --no_mobil \
        --mean_delta_v -12 --target_speeds "5,10,15,20,25,30,34" \
        --distance_range "15,30" --policy_frequency 5 --duration 50 --steps 2000000

    uv run python crash_type_reward/master/train_sb3.py --target side-swipe-left --steps 500000

    uv run python crash_type_reward/master/train_sb3.py --target side-swipe-right --steps 500000

    # Warm-start from an earlier checkpoint
    uv run python crash_type_reward/master/train_sb3.py --target rear-end \
        --warm_start new_results/master/master_rear-end_500k.zip --steps 750000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import highway_env  # noqa: F401  registers crash-v0

from dqn_crasher.utils.config import load_pkg_yaml
from eval_utils import (
    ALL_SPAWN_CONFIGS,
    COMPATIBLE_CONFIGS_BY_TARGET,
    MASTER_ENV_CONFIG,
    REAR_END_TRAINING_DEFAULTS,
    run_stress_eval,
)
from sat_reward_wrapper import (
    MasterRewardWrapper,
    VALID_CRASH_TYPES,
    R_MATCH,
    R_WRONG,
    R_WRONG_RE,
    W_SHAPING,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class SingleAgentNPCWrapper(gym.Wrapper):
    """
    Flattens the multi-agent crash-v0 tuple spaces into scalar spaces for SB3.

    NPC (vehicles[0]) is the DQN-controlled agent. Ego (vehicles[1]) drives
    with IDM/MOBIL internally — no action needed from us.
    """

    def __init__(self, env, frame_stack=5, n_features=5):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.n_features = n_features

        obs_space = env.observation_space[0]
        flat_dim = int(np.prod(obs_space.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.action_space = env.action_space[0]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs[0]), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step((int(action),))
        return self._flatten_obs(obs[0]), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        n_cars = obs.shape[0]
        if self.frame_stack > 1:
            reshaped = obs.reshape((n_cars, self.n_features, self.frame_stack))
            interleaved = np.empty((n_cars * self.frame_stack, self.n_features), dtype=obs.dtype)
            for i in range(n_cars):
                interleaved[i::n_cars] = reshaped[i]
            return interleaved.flatten()
        else:
            return obs.flatten()


class DistanceRandomizerWrapper(gym.Wrapper):
    """Resamples mean_distance uniformly from [d_min, d_max] before each episode reset."""

    def __init__(self, env, d_min: float, d_max: float):
        super().__init__(env)
        self.d_min = d_min
        self.d_max = d_max

    def reset(self, **kwargs):
        d = float(np.random.uniform(self.d_min, self.d_max))
        self.env.unwrapped.config["mean_distance"] = d
        return self.env.reset(**kwargs)


class DeltaVRandomizerWrapper(gym.Wrapper):
    """Resamples mean_delta_v uniformly from [dv_min, dv_max] before each episode reset."""

    def __init__(self, env, dv_min: float, dv_max: float):
        super().__init__(env)
        self.dv_min = dv_min
        self.dv_max = dv_max

    def reset(self, **kwargs):
        dv = float(np.random.uniform(self.dv_min, self.dv_max))
        self.env.unwrapped.config["mean_delta_v"] = dv
        return self.env.reset(**kwargs)


class CrashLoggingCallback(BaseCallback):
    def __init__(
        self,
        target_crash_type,
        log_interval=1000,
        best_model_path=None,
        eval_interval=50_000,
        eval_episodes_per_cell=30,
        eval_mean_distance=20.0,
        eval_mode="compatible",
        eval_distance_grid=None,
        eval_delta_v_grid=None,
        frame_stack=5,
        ego_type="fixed",
        no_mobil=False,
        target_speeds=None,
        policy_frequency=None,
        duration=None,
        verbose=0,
    ):
        super().__init__(verbose)
        self.target_crash_type = target_crash_type
        self.log_interval = log_interval
        self.best_model_path = best_model_path
        self.eval_interval = eval_interval
        self.eval_episodes_per_cell = eval_episodes_per_cell
        self.eval_mean_distance = eval_mean_distance
        self.eval_mode = eval_mode
        self.eval_distance_grid = eval_distance_grid
        self.eval_delta_v_grid = eval_delta_v_grid
        self.frame_stack = frame_stack
        self.ego_type = ego_type
        self.no_mobil = no_mobil
        self.target_speeds = target_speeds
        self.policy_frequency = policy_frequency
        self.duration = duration
        self.crashes = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_crash_types = []
        self._current_ep_reward = 0.0
        self._current_ep_len = 0
        self._best_eval_target_rate = -1.0
        self._last_eval_step = 0

    def _run_selection_eval(self) -> tuple[float, float, int]:
        """Deterministic checkpoint-selection eval on compatible spawn cells."""
        if self.eval_mode == "all":
            spawn_configs = list(ALL_SPAWN_CONFIGS)
        else:
            spawn_configs = COMPATIBLE_CONFIGS_BY_TARGET.get(
                self.target_crash_type,
                ["forward_left", "forward_right"],
            )
        cells = run_stress_eval(
            model=self.model,
            target_crash_type=self.target_crash_type,
            model_path="in_train",
            episodes_per_cell=self.eval_episodes_per_cell,
            spawn_configs=spawn_configs,
            distance_grid=self.eval_distance_grid,
            delta_v_grid=self.eval_delta_v_grid,
            distance_override=self.eval_mean_distance if self.eval_distance_grid is None else None,
            seed=1234,
            frame_stack=self.frame_stack,
            ego_type=self.ego_type,
            no_mobil=self.no_mobil,
            target_speeds=self.target_speeds,
            policy_frequency=self.policy_frequency,
            duration=self.duration,
        )
        total_eps = sum(c.n_episodes for c in cells)
        total_target = sum(c.n_target_hit for c in cells)
        total_crash = sum(c.n_crashed for c in cells)
        target_rate = (total_target / total_eps) if total_eps else 0.0
        crash_rate = (total_crash / total_eps) if total_eps else 0.0
        return target_rate, crash_rate, total_eps

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [])

        self._current_ep_reward += float(rewards[0])
        self._current_ep_len += 1

        for info in infos:
            if dones[0]:
                crashed = info.get("crashed", False)
                self.crashes.append(int(crashed))
                if crashed:
                    outcome = info.get("master_terminal_outcome", "crashed_unknown")
                    ct = info.get("master_crash_type", outcome)
                else:
                    ct = "no_crash"
                self.episode_rewards.append(self._current_ep_reward)
                self.episode_lengths.append(self._current_ep_len)
                self.episode_crash_types.append(ct)
                self._current_ep_reward = 0.0
                self._current_ep_len = 0

        if self.num_timesteps % self.log_interval == 0 and len(self.crashes) > 0:
            window = min(100, len(self.crashes))
            recent_crashes = self.crashes[-window:]
            crash_rate = sum(recent_crashes) / len(recent_crashes)
            recent_types = self.episode_crash_types[-window:]
            target_hits = sum(
                1 for t in recent_types if t == self.target_crash_type
            )
            target_rate = target_hits / len(recent_types)
            wrong_crashes = sum(
                1 for t in recent_types if t not in ("no_crash", self.target_crash_type)
            )
            wrong_rate = wrong_crashes / len(recent_types)
            target_given_crash = target_hits / max(sum(recent_crashes), 1)
            wrong_given_crash = wrong_crashes / max(sum(recent_crashes), 1)

            best_marker = ""
            run_eval = (
                self.best_model_path is not None
                and self.eval_interval > 0
                and (self.num_timesteps - self._last_eval_step) >= self.eval_interval
            )
            if run_eval:
                self._last_eval_step = self.num_timesteps
                eval_target_rate, eval_crash_rate, eval_eps = self._run_selection_eval()
                print(
                    f"    selection_eval: target={eval_target_rate:.1%} "
                    f"crash={eval_crash_rate:.1%} eps={eval_eps}"
                )
                if eval_target_rate > self._best_eval_target_rate:
                    self._best_eval_target_rate = eval_target_rate
                    self.model.save(self.best_model_path)
                    best_marker = " [BEST_BY_EVAL]"

            print(
                f"  Step {self.num_timesteps:>7}: "
                f"crash={crash_rate:.1%}  "
                f"target({self.target_crash_type})={target_rate:.1%}  "
                f"wrong_type={wrong_rate:.1%}  "
                f"target|crash={target_given_crash:.1%}  "
                f"wrong|crash={wrong_given_crash:.1%}  "
                f"(last {window} eps){best_marker}"
            )
        return True

    def print_summary(self):
        from collections import Counter
        all_crash_types = [t for t in self.episode_crash_types if t != "no_crash"]
        if not all_crash_types:
            print("  No crashes recorded.")
            return
        counts = Counter(all_crash_types)
        total = len(all_crash_types)
        target_n = counts.get(self.target_crash_type, 0)
        print(f"\n{'='*50}")
        print(f"  Crash type distribution ({total} total crashes):")
        for ct, n in sorted(counts.items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if ct == self.target_crash_type else ""
            print(f"    {ct:25s}: {n:5d} ({n/total:.1%}){marker}")
        print(f"  Target hit rate: {target_n}/{total} = {target_n/total:.1%}")
        print(f"{'='*50}\n")


_EGO_TYPES = {
    "fixed":       "master_ego_vehicle.MasterEgoVehicle",
    "constrained_mobil": "master_ego_vehicle.ConstrainedMobilEgoVehicle",
    "no_mobil":    "master_ego_vehicle.NoMobilEgoVehicle",
    "randomized":  "randomized_ego_vehicle.RandomizedEgoVehicle",
    "transfer_randomized": "randomized_ego_vehicle.TransferRandomizedEgoVehicle",
    "cautious":    "randomized_ego_vehicle.CautiousEgoVehicle",
    "aggressive":  "randomized_ego_vehicle.AggressiveEgoVehicle",
}


def make_env(
    target_crash_type: str,
    frame_stack: int = 5,
    spawn_subset=None,
    mean_distance: float | None = None,
    mean_delta_v: float | None = None,
    initial_lane_id: int | None = None,
    distance_range: tuple[float, float] | None = None,
    delta_v_range: tuple[float, float] | None = None,
    ego_type: str = "fixed",
    no_mobil: bool = False,
    target_speeds: list[float] | None = None,
    policy_frequency: int | None = None,
    duration: int | None = None,
) -> gym.Env:
    # --no_mobil forces ego to NoMobilEgoVehicle (same IDM params, change_lane_policy no-op).
    # This is the correct mechanism — use_mobil config flag only applies to the dual-vehicle
    # spawn path (controlled_vehicles=2), not our single_controlled_vehicle_spawn path.
    resolved_ego = "no_mobil" if no_mobil else ego_type

    gym_config = load_pkg_yaml(MASTER_ENV_CONFIG)
    gym_config["observation"]["observation_config"]["frame_stack"] = frame_stack
    gym_config["other_vehicles_type"] = _EGO_TYPES[resolved_ego]

    if spawn_subset is not None:
        yaml_configs = set(gym_config["spawn_configs"])
        unknown = [c for c in spawn_subset if c not in yaml_configs]
        if unknown:
            raise ValueError(
                f"--spawn_subset contains configs not in the YAML pool: {unknown}. "
                f"Valid: {sorted(yaml_configs)}"
            )
        gym_config["spawn_configs"] = list(spawn_subset)

    if mean_distance is not None:
        gym_config["mean_distance"] = float(mean_distance)
    if mean_delta_v is not None:
        gym_config["mean_delta_v"] = float(mean_delta_v)
    if initial_lane_id is not None:
        gym_config["initial_lane_id"] = int(initial_lane_id)
    if target_speeds is not None:
        gym_config["action"]["action_config"]["target_speeds"] = list(target_speeds)
    if policy_frequency is not None:
        gym_config["policy_frequency"] = int(policy_frequency)
    if duration is not None:
        gym_config["duration"] = int(duration)

    env = gym.make("crash-v0", config=gym_config)
    env = MasterRewardWrapper(env, target_crash_type=target_crash_type)
    if distance_range is not None:
        env = DistanceRandomizerWrapper(env, d_min=distance_range[0], d_max=distance_range[1])
    if delta_v_range is not None:
        env = DeltaVRandomizerWrapper(env, dv_min=delta_v_range[0], dv_max=delta_v_range[1])
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
    fig.suptitle(f"Master NPC DQN — target: {target_crash_type} ({total_timesteps//1000}k steps)")

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
    plot_path = os.path.join(
        results_dir, f"training_curves_{target_crash_type}_{total_timesteps//1000}k.png"
    )
    plt.savefig(plot_path, dpi=150)
    print(f"Plots saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", required=True, choices=sorted(VALID_CRASH_TYPES),
        help="Crash type the NPC should engineer the ego to produce",
    )
    parser.add_argument("--steps", type=int, default=750_000)
    parser.add_argument(
        "--spawn_subset", default=None,
        help="Comma-separated subset of spawn configs "
             "(e.g. 'forward_left,forward_right' for rear-end cut-in training)",
    )
    parser.add_argument(
        "--warm_start", default=None,
        help="Path to a .zip checkpoint to continue training from",
    )
    parser.add_argument(
        "--warm_start_eps", type=float, default=0.1,
        help="Initial exploration epsilon for warm-start runs (default 0.1)",
    )
    parser.add_argument(
        "--mean_distance", type=float, default=None,
        help="Override spawn mean_distance (m). Default is YAML value (20 m).",
    )
    parser.add_argument(
        "--mean_delta_v", type=float, default=None,
        help="Override spawn mean_delta_v (m/s). Applied to NPC speed: negative → NPC slower → ego closes from behind (desired for rear-end cut-in). Default std=5 from use_spawn_distribution.",
    )
    parser.add_argument(
        "--initial_lane_id", type=int, default=None,
        help="Force ego to start in this lane index (0=rightmost, 2=leftmost in 3-lane road).",
    )
    parser.add_argument(
        "--distance_range", default=None,
        help="Randomize spawn distance uniformly from this range each episode, e.g. '5,30'.",
    )
    parser.add_argument(
        "--ego_type", default="fixed",
        choices=list(_EGO_TYPES),
        help="Ego vehicle type: 'fixed'=MasterEgoVehicle (default); "
             "'randomized'=domain randomization over IDM/MOBIL params each episode; "
             "'cautious'/'aggressive'=fixed eval profiles.",
    )
    parser.add_argument(
        "--no_mobil", action="store_true",
        help="Disable MOBIL lane-change policy for the ego vehicle. "
             "Required for rear-end training: prevents ego from escaping longitudinal "
             "collision by changing lanes when NPC decelerates.",
    )
    parser.add_argument(
        "--target_speeds", default=None,
        help="Comma-separated NPC target speeds, e.g. '5,10,15,20,25,30,34'. "
             "Overrides the YAML action_config.target_speeds. Use with --no_mobil for "
             "rear-end: larger steps create unsafe closing speeds at greater distances.",
    )
    parser.add_argument(
        "--policy_frequency", type=int, default=None,
        help="Override policy_frequency (Hz). Default=YAML (1 Hz). "
             "Use 5 for rear-end: step_size=2.4m < crash window (~5m) → distance-invariant.",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Override episode duration in steps. Use 50 with --policy_frequency 5 (= 10 real seconds).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for torch/numpy/random (for reproducible runs and parallel seed search).",
    )
    parser.add_argument(
        "--apply_rear_end_defaults", action="store_true",
        help="Apply REAR_END_TRAINING_DEFAULTS (cut-in spawns, no_mobil, delta_v=-12, "
             "distance_range 15-30m, policy_frequency=5, etc.). See eval_utils.py.",
    )
    parser.add_argument(
        "--selection_eval_interval", type=int, default=50_000,
        help="Run deterministic compatible-cell eval every N env steps for checkpoint selection.",
    )
    parser.add_argument(
        "--selection_eval_episodes", type=int, default=30,
        help="Episodes per compatible spawn cell during checkpoint-selection eval.",
    )
    parser.add_argument(
        "--selection_eval_distance", type=float, default=20.0,
        help="Mean distance used during checkpoint-selection eval.",
    )
    parser.add_argument(
        "--selection_eval_mode", choices=["compatible", "all"], default="all",
        help="Checkpoint selection eval scope: compatible-only or all spawns.",
    )
    parser.add_argument(
        "--selection_eval_distance_grid", default=None,
        help="Optional comma-separated distance grid for selection eval (e.g. '10,20,30').",
    )
    parser.add_argument(
        "--selection_eval_delta_v_grid", default=None,
        help="Optional comma-separated delta-v grid for selection eval (e.g. '-16,-12,-8,-4,0').",
    )
    parser.add_argument(
        "--rear_end_two_stage_curriculum", action="store_true",
        help="For rear-end only: stage1 compatible cut-in training, then stage2 all-spawn + delta-v randomization.",
    )
    parser.add_argument(
        "--rear_end_mobil_ego_transfer", action="store_true",
        help="Enable MOBIL-on ego transfer curriculum: stage1 constrained MOBIL ego, "
             "stage2 transfer-randomized MOBIL ego (no no_mobil).",
    )
    parser.add_argument(
        "--stage1_ego_type", default=None,
        help="Optional ego type override for stage 1 when two-stage curriculum is enabled.",
    )
    parser.add_argument(
        "--stage2_ego_type", default=None,
        help="Optional ego type override for stage 2 when two-stage curriculum is enabled.",
    )
    parser.add_argument(
        "--stage1_fraction", type=float, default=0.5,
        help="Fraction of total steps used in stage1 when --rear_end_two_stage_curriculum is enabled.",
    )
    parser.add_argument(
        "--stage2_distance_range", default="10,40",
        help="Stage2 distance randomization range, e.g. '10,40'.",
    )
    parser.add_argument(
        "--stage2_delta_v_range", default="-16,4",
        help="Stage2 delta-v randomization range, e.g. '-16,4'.",
    )
    args = parser.parse_args()

    if args.apply_rear_end_defaults:
        if args.target != "rear-end":
            raise SystemExit("--apply_rear_end_defaults requires --target rear-end")
        d = REAR_END_TRAINING_DEFAULTS
        if args.spawn_subset is None:
            args.spawn_subset = d["spawn_subset"]
        if not args.no_mobil:
            args.no_mobil = d["no_mobil"]
        if args.mean_delta_v is None:
            args.mean_delta_v = d["mean_delta_v"]
        if args.distance_range is None:
            args.distance_range = d["distance_range"]
        if args.target_speeds is None:
            args.target_speeds = d["target_speeds"]
        if args.policy_frequency is None:
            args.policy_frequency = d["policy_frequency"]
        if args.duration is None:
            args.duration = d["duration"]

    if args.rear_end_mobil_ego_transfer:
        if args.target != "rear-end":
            raise SystemExit("--rear_end_mobil_ego_transfer requires --target rear-end")
        d = REAR_END_TRAINING_DEFAULTS
        # Keep MOBIL enabled throughout transfer.
        args.no_mobil = False
        args.rear_end_two_stage_curriculum = True
        if args.spawn_subset is None:
            args.spawn_subset = d["spawn_subset"]
        if args.mean_delta_v is None:
            args.mean_delta_v = d["mean_delta_v"]
        if args.distance_range is None:
            args.distance_range = d["distance_range"]
        if args.target_speeds is None:
            args.target_speeds = d["target_speeds"]
        if args.policy_frequency is None:
            args.policy_frequency = d["policy_frequency"]
        if args.duration is None:
            args.duration = d["duration"]
        if args.selection_eval_mode == "compatible":
            # Robustness is the objective for transfer.
            args.selection_eval_mode = "all"
        if args.selection_eval_distance_grid is None:
            args.selection_eval_distance_grid = "10,20,30"
        if args.selection_eval_delta_v_grid is None:
            args.selection_eval_delta_v_grid = "-16,-12,-8,-4,0,4"
        if args.stage1_ego_type is None:
            args.stage1_ego_type = "constrained_mobil"
        if args.stage2_ego_type is None:
            args.stage2_ego_type = "transfer_randomized"

    if args.stage1_ego_type is not None and args.stage1_ego_type not in _EGO_TYPES:
        raise SystemExit(f"--stage1_ego_type must be one of {list(_EGO_TYPES)}")
    if args.stage2_ego_type is not None and args.stage2_ego_type not in _EGO_TYPES:
        raise SystemExit(f"--stage2_ego_type must be one of {list(_EGO_TYPES)}")

    frame_stack = 5
    total_timesteps = args.steps
    target_crash_type = args.target
    spawn_subset = (
        [s.strip() for s in args.spawn_subset.split(",")]
        if args.spawn_subset else None
    )

    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "new_results", "master"
    )
    os.makedirs(results_dir, exist_ok=True)

    stage1_ego_type = args.stage1_ego_type or args.ego_type
    stage2_ego_type = args.stage2_ego_type or args.ego_type

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    print("=" * 60)
    print(f"Master NPC DQN Training — target: {target_crash_type}")
    print("=" * 60)
    print(f"Total timesteps : {total_timesteps}")
    print(f"Spawn subset    : {spawn_subset or 'all (random)'}")
    distance_range = None
    if args.distance_range:
        parts = args.distance_range.split(",")
        distance_range = (float(parts[0].strip()), float(parts[1].strip()))
    delta_v_range = None

    if distance_range is not None:
        print(f"Distance range  : {distance_range[0]}-{distance_range[1]}m (uniform per episode)")
    else:
        print(f"Mean distance   : {args.mean_distance or 'YAML default (20m)'}")
    print(f"Mean delta-v    : {args.mean_delta_v if args.mean_delta_v is not None else 'YAML default (0)'}")
    print(f"Ego type        : {args.ego_type}")
    print(f"No MOBIL        : {args.no_mobil}")
    if args.rear_end_two_stage_curriculum:
        print(f"Stage1 ego      : {stage1_ego_type}")
        print(f"Stage2 ego      : {stage2_ego_type}")
    target_speeds_list = (
        [float(x) for x in args.target_speeds.split(",")]
        if args.target_speeds else None
    )
    if target_speeds_list is not None:
        print(f"Target speeds   : {target_speeds_list}")
    if args.policy_frequency is not None:
        print(f"Policy freq     : {args.policy_frequency} Hz")
    if args.duration is not None:
        print(f"Episode duration: {args.duration} steps")
    print(
        f"Reward          : +{R_MATCH} (match) / {R_WRONG_RE} (RE wrong type) / "
        f"-{R_WRONG} (SSL/SSR wrong) + shaping w={W_SHAPING}"
    )
    if args.warm_start:
        print(f"Warm start      : {args.warm_start}")
    print()

    env = make_env(
        target_crash_type=target_crash_type,
        frame_stack=frame_stack,
        spawn_subset=spawn_subset,
        mean_distance=args.mean_distance,
        mean_delta_v=args.mean_delta_v,
        initial_lane_id=args.initial_lane_id,
        distance_range=distance_range,
        ego_type=stage1_ego_type,
        no_mobil=args.no_mobil,
        target_speeds=target_speeds_list,
        policy_frequency=args.policy_frequency,
        duration=args.duration,
    )

    seed_tag = f"_s{args.seed}" if args.seed is not None else ""
    best_model_path = os.path.join(
        results_dir, f"master_{target_crash_type}_{total_timesteps//1000}k{seed_tag}_best"
    )
    callback = CrashLoggingCallback(
        target_crash_type=target_crash_type,
        log_interval=10000,
        best_model_path=best_model_path,
        eval_interval=args.selection_eval_interval,
        eval_episodes_per_cell=args.selection_eval_episodes,
        eval_mean_distance=args.selection_eval_distance,
        eval_mode=args.selection_eval_mode,
        eval_distance_grid=(
            [float(x.strip()) for x in args.selection_eval_distance_grid.split(",")]
            if args.selection_eval_distance_grid else None
        ),
        eval_delta_v_grid=(
            [float(x.strip()) for x in args.selection_eval_delta_v_grid.split(",")]
            if args.selection_eval_delta_v_grid else None
        ),
        frame_stack=frame_stack,
        ego_type=stage1_ego_type,
        no_mobil=args.no_mobil,
        target_speeds=target_speeds_list,
        policy_frequency=args.policy_frequency,
        duration=args.duration,
    )

    if args.warm_start:
        model = DQN.load(args.warm_start, env=env)
        model.exploration_initial_eps = args.warm_start_eps
        model.exploration_final_eps   = 0.05
        model.exploration_fraction    = 0.3
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

    if args.rear_end_two_stage_curriculum:
        if target_crash_type != "rear-end":
            raise SystemExit("--rear_end_two_stage_curriculum requires --target rear-end")
        if not (0.0 < args.stage1_fraction < 1.0):
            raise SystemExit("--stage1_fraction must be in (0,1)")

        stage1_steps = int(total_timesteps * args.stage1_fraction)
        stage2_steps = total_timesteps - stage1_steps
        s2_dist = tuple(float(x.strip()) for x in args.stage2_distance_range.split(","))
        s2_dv = tuple(float(x.strip()) for x in args.stage2_delta_v_range.split(","))

        print("Starting stage 1 (compatible cut-in curriculum)...")
        model.learn(
            total_timesteps=stage1_steps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=args.warm_start is None,
        )

        print("Starting stage 2 (all-spawn + speed/domain randomization)...")
        env_stage2 = make_env(
            target_crash_type=target_crash_type,
            frame_stack=frame_stack,
            spawn_subset=None,  # all spawn configs from YAML
            mean_distance=args.mean_distance,
            mean_delta_v=args.mean_delta_v,
            initial_lane_id=args.initial_lane_id,
            distance_range=s2_dist,
            delta_v_range=s2_dv,
            ego_type=stage2_ego_type,
            no_mobil=args.no_mobil,
            target_speeds=target_speeds_list,
            policy_frequency=args.policy_frequency,
            duration=args.duration,
        )
        callback.ego_type = stage2_ego_type
        model.set_env(env_stage2)
        model.learn(
            total_timesteps=stage2_steps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    else:
        print("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=args.warm_start is None,
        )

    model_path = os.path.join(
        results_dir, f"master_{target_crash_type}_{total_timesteps//1000}k{seed_tag}"
    )
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    callback.print_summary()
    plot_training(callback, results_dir, target_crash_type, total_timesteps)

    env.close()


if __name__ == "__main__":
    main()
