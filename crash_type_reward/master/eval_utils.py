"""
Shared eval helpers for master NPC SB3 evaluation.

Spawn config geometry (NPC relative to ego):
  forward_left/right : NPC ahead in adjacent lane  → natural cut-in for rear-end
  adjacent_left/right: NPC alongside in adjacent lane
  forward_center     : NPC ahead in same lane
  behind_*           : NPC behind ego
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import numpy as np

from dqn_crasher.utils.config import load_pkg_yaml
from sat_reward_wrapper import MasterRewardWrapper, VALID_CRASH_TYPES
from stable_baselines3.common.monitor import Monitor


ALL_SPAWN_CONFIGS = [
    "behind_left", "behind_right", "behind_center",
    "adjacent_left", "adjacent_right",
    "forward_left", "forward_right", "forward_center",
]

# Geometrically natural spawn positions (NPC starts favourably for the target).
COMPATIBLE_CONFIGS_BY_TARGET = {
    "rear-end":         ["forward_left", "forward_right"],
    "side-swipe-left":  ["adjacent_left",  "forward_left"],
    "side-swipe-right": ["adjacent_right", "forward_right"],
}

# Structurally infeasible or systematically wrong-type for the target.
INFEASIBLE_CELLS_BY_TARGET = {
    "rear-end":         {"behind_center", "behind_left", "behind_right",
                         "adjacent_left", "adjacent_right"},
    "side-swipe-left":  {"adjacent_right", "behind_right", "forward_right"},
    "side-swipe-right": {"adjacent_left",  "behind_left",  "forward_left"},
}

# Eval grid modes (see get_spawn_configs_for_eval_mode).
EVAL_MODES = ("compatible", "feasible", "all")

# Master pipeline env YAML (ego-caused crashes); legacy sb3 uses crash_type_reward.yaml.
MASTER_ENV_CONFIG = "configs/env/crash_type_reward_master.yaml"

# Recommended CLI knobs for master rear-end cut-in curriculum (train / eval / render).
REAR_END_TRAINING_DEFAULTS = {
    "spawn_subset": "forward_left,forward_right",
    "no_mobil": True,
    "mean_delta_v": -12.0,
    "distance_range": "15,30",
    "target_speeds": "5,10,15,20,25,30,34",
    "policy_frequency": 5,
    "duration": 50,
    "ego_type": "no_mobil",
}


def get_spawn_configs_for_eval_mode(target_crash_type: str, eval_mode: str) -> list[str]:
    """Return spawn configs for an eval protocol."""
    assert eval_mode in EVAL_MODES, f"eval_mode must be one of {EVAL_MODES}"
    if eval_mode == "all":
        return list(ALL_SPAWN_CONFIGS)
    if eval_mode == "compatible":
        return list(COMPATIBLE_CONFIGS_BY_TARGET.get(target_crash_type, ALL_SPAWN_CONFIGS))
    infeasible = INFEASIBLE_CELLS_BY_TARGET.get(target_crash_type, set())
    return [c for c in ALL_SPAWN_CONFIGS if c not in infeasible]


_EGO_TYPES = {
    "fixed":       "master_ego_vehicle.MasterEgoVehicle",
    "constrained_mobil": "master_ego_vehicle.ConstrainedMobilEgoVehicle",
    "no_mobil":    "master_ego_vehicle.NoMobilEgoVehicle",
    "randomized":  "randomized_ego_vehicle.RandomizedEgoVehicle",
    "transfer_randomized": "randomized_ego_vehicle.TransferRandomizedEgoVehicle",
    "cautious":    "randomized_ego_vehicle.CautiousEgoVehicle",
    "aggressive":  "randomized_ego_vehicle.AggressiveEgoVehicle",
}


def build_env(
    target_crash_type: str,
    *,
    frame_stack: int = 5,
    spawn_configs: Optional[list] = None,
    mean_distance: Optional[float] = None,
    mean_delta_v: Optional[float] = None,
    use_spawn_distribution: Optional[bool] = None,
    initial_lane_id: Optional[int] = None,
    ego_type: str = "fixed",
    no_mobil: bool = False,
    target_speeds: Optional[list] = None,
    policy_frequency: Optional[int] = None,
    duration: Optional[int] = None,
    render_mode: Optional[str] = None,
):
    import gymnasium as gym
    import highway_env  # noqa: F401
    from train_sb3 import SingleAgentNPCWrapper

    assert target_crash_type in VALID_CRASH_TYPES, target_crash_type
    assert ego_type in _EGO_TYPES, f"ego_type must be one of {list(_EGO_TYPES)}"

    resolved_ego = "no_mobil" if no_mobil else ego_type

    gym_config = load_pkg_yaml(MASTER_ENV_CONFIG)
    gym_config["observation"]["observation_config"]["frame_stack"] = frame_stack
    gym_config["other_vehicles_type"] = _EGO_TYPES[resolved_ego]

    if spawn_configs is not None:
        gym_config["spawn_configs"] = list(spawn_configs)
    if mean_distance is not None:
        gym_config["mean_distance"] = float(mean_distance)
    if mean_delta_v is not None:
        gym_config["mean_delta_v"] = float(mean_delta_v)
    if use_spawn_distribution is not None:
        gym_config["use_spawn_distribution"] = bool(use_spawn_distribution)
    if initial_lane_id is not None:
        gym_config["initial_lane_id"] = int(initial_lane_id)
    if target_speeds is not None:
        gym_config["action"]["action_config"]["target_speeds"] = list(target_speeds)
    if policy_frequency is not None:
        gym_config["policy_frequency"] = int(policy_frequency)
    if duration is not None:
        gym_config["duration"] = int(duration)

    make_kwargs = {"config": gym_config}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    env = gym.make("crash-v0", **make_kwargs)
    env = MasterRewardWrapper(env, target_crash_type=target_crash_type)
    env = SingleAgentNPCWrapper(env, frame_stack=frame_stack)
    env = Monitor(env)
    return env


@dataclass
class CellResult:
    model_path: str
    target_crash_type: str
    spawn_config: str
    mean_distance: Optional[float]
    mean_delta_v: Optional[float]
    use_spawn_distribution: Optional[bool]
    n_episodes: int
    n_crashed: int
    n_target_hit: int
    n_crashed_unknown: int
    mean_episode_length: float
    observed_crash_type_counts: dict = field(default_factory=dict)

    @property
    def target_hit_rate(self) -> float:
        return self.n_target_hit / self.n_episodes if self.n_episodes else 0.0

    @property
    def any_crash_rate(self) -> float:
        return self.n_crashed / self.n_episodes if self.n_episodes else 0.0

    @property
    def wrong_type_crash_rate(self) -> float:
        wrong = self.n_crashed - self.n_target_hit - self.n_crashed_unknown
        return max(0, wrong) / self.n_episodes if self.n_episodes else 0.0

    def as_row(self) -> dict:
        return {
            "model_path": self.model_path,
            "target": self.target_crash_type,
            "spawn_config": self.spawn_config,
            "mean_distance": self.mean_distance if self.mean_distance is not None else "",
            "mean_delta_v": self.mean_delta_v if self.mean_delta_v is not None else "",
            "use_spawn_distribution": (
                self.use_spawn_distribution if self.use_spawn_distribution is not None else ""
            ),
            "n_episodes": self.n_episodes,
            "n_crashed": self.n_crashed,
            "n_target_hit": self.n_target_hit,
            "n_crashed_unknown": self.n_crashed_unknown,
            "target_hit_rate": f"{self.target_hit_rate:.4f}",
            "any_crash_rate": f"{self.any_crash_rate:.4f}",
            "wrong_type_crash_rate": f"{self.wrong_type_crash_rate:.4f}",
            "mean_episode_length": f"{self.mean_episode_length:.2f}",
            "observed_crash_types": ",".join(
                f"{k}:{v}" for k, v in sorted(self.observed_crash_type_counts.items())
            ),
        }


def _run_cell(
    model,
    model_path: str,
    target_crash_type: str,
    spawn_config: str,
    episodes_per_cell: int,
    seed_base: int,
    *,
    mean_distance: Optional[float] = None,
    mean_delta_v: Optional[float] = None,
    use_spawn_distribution: Optional[bool] = None,
    initial_lane_id: Optional[int] = None,
    frame_stack: int = 5,
    ego_type: str = "fixed",
    no_mobil: bool = False,
    target_speeds: Optional[list] = None,
    policy_frequency: Optional[int] = None,
    duration: Optional[int] = None,
) -> CellResult:
    env = build_env(
        target_crash_type,
        frame_stack=frame_stack,
        spawn_configs=[spawn_config],
        mean_distance=mean_distance,
        mean_delta_v=mean_delta_v,
        use_spawn_distribution=use_spawn_distribution,
        initial_lane_id=initial_lane_id,
        ego_type=ego_type,
        no_mobil=no_mobil,
        target_speeds=target_speeds,
        policy_frequency=policy_frequency,
        duration=duration,
    )
    crash_type_counts: dict = {}
    n_crashed = n_target_hit = n_crashed_unknown = 0
    ep_lengths: list[int] = []

    try:
        for ep_idx in range(episodes_per_cell):
            obs, _ = env.reset(seed=seed_base + ep_idx)
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                ep_len += 1
            ep_lengths.append(ep_len)
            if info.get("crashed", False):
                n_crashed += 1
                ct = info.get("master_crash_type")
                if ct is None:
                    n_crashed_unknown += 1
                    ct = "crashed_unknown"
                crash_type_counts[ct] = crash_type_counts.get(ct, 0) + 1
                if ct == target_crash_type:
                    n_target_hit += 1
    finally:
        env.close()

    return CellResult(
        model_path=model_path,
        target_crash_type=target_crash_type,
        spawn_config=spawn_config,
        mean_distance=mean_distance,
        mean_delta_v=mean_delta_v,
        use_spawn_distribution=use_spawn_distribution,
        n_episodes=episodes_per_cell,
        n_crashed=n_crashed,
        n_target_hit=n_target_hit,
        n_crashed_unknown=n_crashed_unknown,
        mean_episode_length=float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        observed_crash_type_counts=crash_type_counts,
    )


def run_stress_eval(
    model,
    target_crash_type: str,
    *,
    model_path: str = "",
    episodes_per_cell: int = 40,
    spawn_configs: Optional[Iterable[str]] = None,
    distance_grid: Optional[Iterable[float]] = None,
    delta_v_grid: Optional[Iterable[float]] = None,
    distance_override: Optional[float] = None,
    initial_lane_id: Optional[int] = None,
    seed: int = 0,
    frame_stack: int = 5,
    ego_type: str = "fixed",
    no_mobil: bool = False,
    target_speeds: Optional[list] = None,
    policy_frequency: Optional[int] = None,
    duration: Optional[int] = None,
    on_cell_done=None,
) -> list[CellResult]:
    configs = list(spawn_configs) if spawn_configs is not None else list(ALL_SPAWN_CONFIGS)
    distances = [distance_override] if distance_grid is None else list(distance_grid)
    deltas = [None] if delta_v_grid is None else list(delta_v_grid)
    use_dist = None if distance_grid is None and delta_v_grid is None else True

    results: list[CellResult] = []
    cell_idx = 0
    for cfg in configs:
        for d in distances:
            for dv in deltas:
                cell_seed = seed + cell_idx * 10_000
                cell_idx += 1
                result = _run_cell(
                    model=model,
                    model_path=model_path,
                    target_crash_type=target_crash_type,
                    spawn_config=cfg,
                    episodes_per_cell=episodes_per_cell,
                    seed_base=cell_seed,
                    mean_distance=d,
                    mean_delta_v=dv,
                    use_spawn_distribution=use_dist,
                    initial_lane_id=initial_lane_id,
                    frame_stack=frame_stack,
                    ego_type=ego_type,
                    no_mobil=no_mobil,
                    target_speeds=target_speeds,
                    policy_frequency=policy_frequency,
                    duration=duration,
                )
                results.append(result)
                if on_cell_done is not None:
                    on_cell_done(result)
    return results


def summary_metrics(results: list[CellResult], target_crash_type: str) -> dict:
    compat = set(COMPATIBLE_CONFIGS_BY_TARGET.get(target_crash_type, ALL_SPAWN_CONFIGS))
    infeasible = INFEASIBLE_CELLS_BY_TARGET.get(target_crash_type, set())
    compat_rates = [r.target_hit_rate for r in results if r.spawn_config in compat]
    feasible_rates = [r.target_hit_rate for r in results if r.spawn_config not in infeasible]
    all_rates = [r.target_hit_rate for r in results]
    any_rates = [r.any_crash_rate for r in results]
    wrong_rates = [r.wrong_type_crash_rate for r in results]
    return {
        "eval/mean_compatible_target_hit": float(np.mean(compat_rates)) if compat_rates else 0.0,
        "eval/min_compatible_target_hit":  float(np.min(compat_rates))  if compat_rates else 0.0,
        "eval/mean_feasible_target_hit":   float(np.mean(feasible_rates)) if feasible_rates else 0.0,
        "eval/mean_all_configs_target_hit": float(np.mean(all_rates))   if all_rates else 0.0,
        "eval/mean_any_crash_rate":         float(np.mean(any_rates))   if any_rates else 0.0,
        "eval/mean_wrong_type_crash_rate":  float(np.mean(wrong_rates)) if wrong_rates else 0.0,
    }


def format_crash_type_breakdown(result: CellResult) -> str:
    if not result.observed_crash_type_counts:
        return "no crashes"
    parts = [f"{k}={v}" for k, v in sorted(result.observed_crash_type_counts.items())]
    return " ".join(parts)
