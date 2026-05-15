"""
Shared helpers for SB3 SAT-reward evaluation.

Used by:
  - eval_sb3.py (standalone stress-test CLI)
  - train_sb3.py (end-of-train eval for wandb sweeps)

Builds an env with the same wrapper stack as training, and runs a stress
grid over spawn_configs (and optionally mean_distance / mean_delta_v),
returning per-cell target-hit rates.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gymnasium as gym
import highway_env  # noqa: F401  (registers crash-v0)
import numpy as np

from dqn_crasher.utils.config import load_pkg_yaml
from sat_reward_wrapper import SATRewardWrapper, VALID_CRASH_TYPES
from stable_baselines3.common.monitor import Monitor


ALL_SPAWN_CONFIGS = [
    "behind_left", "behind_right", "behind_center",
    "adjacent_left", "adjacent_right",
    "forward_left", "forward_right", "forward_center",
]

COMPATIBLE_CONFIGS_BY_TARGET = {
    "rear-end": ["behind_center", "behind_left", "behind_right"],
    "side-swipe-left": ["adjacent_left", "behind_left"],
    "side-swipe-right": ["adjacent_right", "behind_right"],
}

# Cells that the env geometry makes structurally unreachable for a given target.
# Excluded from the "feasible" aggregate so hyperparameter tuning isn't penalized
# for failing on configs that no policy could succeed at.
#
#   rear-end @ adjacent_*    : crash_env.py:279 hardcodes spawn_distance1=0,
#                              so NPC and victim start at dx=0. No approach
#                              geometry exists — any contact is a side-swipe.
#   side-swipe-left @ *_right: NPC spawns on victim's RIGHT (mirror). Only
#                              side-swipe-right is physically reachable.
#   side-swipe-right @ *_left: mirror of the above.
INFEASIBLE_CELLS_BY_TARGET = {
    "rear-end": {"adjacent_left", "adjacent_right"},
    "side-swipe-left": {"adjacent_right", "behind_right"},
    "side-swipe-right": {"adjacent_left", "behind_left"},
}


def build_env(
    target_crash_type: str,
    *,
    frame_stack: int = 5,
    spawn_configs: Optional[list] = None,
    mean_distance: Optional[float] = None,
    mean_delta_v: Optional[float] = None,
    use_spawn_distribution: Optional[bool] = None,
):
    """
    Build the crash-v0 env with the same wrapper stack as train_sb3.py.

    Any kwarg set to None is left at the YAML default.
    """
    from train_sb3 import SingleAgentNPCWrapper

    assert target_crash_type in VALID_CRASH_TYPES, target_crash_type

    gym_config = load_pkg_yaml("configs/env/crash_type_reward.yaml")
    gym_config["observation"]["observation_config"]["frame_stack"] = frame_stack
    if spawn_configs is not None:
        gym_config["spawn_configs"] = list(spawn_configs)
    if mean_distance is not None:
        gym_config["mean_distance"] = float(mean_distance)
    if mean_delta_v is not None:
        gym_config["mean_delta_v"] = float(mean_delta_v)
    if use_spawn_distribution is not None:
        gym_config["use_spawn_distribution"] = bool(use_spawn_distribution)

    env = gym.make("crash-v0", config=gym_config)
    env = SATRewardWrapper(env, target_crash_type=target_crash_type)
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

    def as_row(self) -> dict:
        return {
            "model_path": self.model_path,
            "target": self.target_crash_type,
            "spawn_config": self.spawn_config,
            "mean_distance": self.mean_distance if self.mean_distance is not None else "",
            "mean_delta_v": self.mean_delta_v if self.mean_delta_v is not None else "",
            "use_spawn_distribution": self.use_spawn_distribution
            if self.use_spawn_distribution is not None else "",
            "n_episodes": self.n_episodes,
            "n_crashed": self.n_crashed,
            "n_target_hit": self.n_target_hit,
            "n_crashed_unknown": self.n_crashed_unknown,
            "target_hit_rate": f"{self.target_hit_rate:.4f}",
            "any_crash_rate": f"{self.any_crash_rate:.4f}",
            "mean_episode_length": f"{self.mean_episode_length:.2f}",
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
    frame_stack: int = 5,
    assert_spawn_config_in_info: bool = True,
) -> CellResult:
    env = build_env(
        target_crash_type,
        frame_stack=frame_stack,
        spawn_configs=[spawn_config],
        mean_distance=mean_distance,
        mean_delta_v=mean_delta_v,
        use_spawn_distribution=use_spawn_distribution,
    )
    crash_type_counts: dict = {}
    n_crashed = n_target_hit = n_crashed_unknown = 0
    ep_lengths: list[int] = []

    try:
        for ep_idx in range(episodes_per_cell):
            obs, info = env.reset(seed=seed_base + ep_idx)
            if assert_spawn_config_in_info and ep_idx == 0:
                assert "spawn_config" in info, (
                    f"info missing 'spawn_config' after reset; wrapper stack may be "
                    f"stripping it. got keys: {list(info)}"
                )
                if info["spawn_config"] != spawn_config:
                    # cold reset before the env has picked from spawn_configs may report stale value
                    pass
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                ep_len += 1
            ep_lengths.append(ep_len)
            if info.get("crashed", False):
                n_crashed += 1
                ct = info.get("sat_crash_type")
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
    seed: int = 0,
    frame_stack: int = 5,
    on_cell_done=None,
) -> list[CellResult]:
    """
    Run per-cell eval and return a list of CellResult.

    If distance_grid / delta_v_grid are None, runs one cell per spawn_config at
    YAML defaults. If set, runs the full cross-product.
    """
    configs = list(spawn_configs) if spawn_configs is not None else list(ALL_SPAWN_CONFIGS)
    distances = [None] if distance_grid is None else list(distance_grid)
    deltas = [None] if delta_v_grid is None else list(delta_v_grid)
    use_dist = None if distance_grid is None and delta_v_grid is None else True

    results: list[CellResult] = []
    cell_idx = 0
    for cfg in configs:
        for d in distances:
            for dv in deltas:
                # distinct seed per cell so grids don't collide
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
                    frame_stack=frame_stack,
                )
                results.append(result)
                if on_cell_done is not None:
                    on_cell_done(result)
    return results


def summary_metrics(results: list[CellResult], target_crash_type: str) -> dict:
    """
    Compute sweep-friendly scalar metrics from a list of per-cell results.
    """
    compat = set(COMPATIBLE_CONFIGS_BY_TARGET.get(target_crash_type, ALL_SPAWN_CONFIGS))
    infeasible = INFEASIBLE_CELLS_BY_TARGET.get(target_crash_type, set())
    compat_rates = [r.target_hit_rate for r in results if r.spawn_config in compat]
    feasible_rates = [r.target_hit_rate for r in results if r.spawn_config not in infeasible]
    all_rates = [r.target_hit_rate for r in results]
    any_rates = [r.any_crash_rate for r in results]
    return {
        "eval/mean_compatible_target_hit": float(np.mean(compat_rates)) if compat_rates else 0.0,
        "eval/min_compatible_target_hit": float(np.min(compat_rates)) if compat_rates else 0.0,
        "eval/mean_feasible_target_hit": float(np.mean(feasible_rates)) if feasible_rates else 0.0,
        "eval/mean_all_configs_target_hit": float(np.mean(all_rates)) if all_rates else 0.0,
        "eval/mean_any_crash_rate": float(np.mean(any_rates)) if any_rates else 0.0,
    }
