"""
Master NPC reward wrapper: NPC engineers scenarios where the ego (IDM/MOBIL)
causes the crash rather than the NPC directly crashing into the ego.

Dense shaping: alignment of NPC→ego vector (in NPC's local frame) with the
target direction, weighted by proximity. Rewards NPC for holding a position
where ego will naturally approach from the target direction.

Terminal: +R_MATCH if ego's crash classification matches target_crash_type,
          R_WRONG_RE (<0) for wrong crash type when targeting rear-end (prevents
          side-swipe / rear-ended shortcuts). SSL/SSR keep R_WRONG=0.

Convention:
  vehicles[0] = NPC adversary (DQN-controlled)
  vehicles[1] = ego (IDM/MOBIL, the vehicle being manipulated)

NPC→ego vector in NPC's local frame (x=forward, y=left):
  rear-end         target_dir = (-1,  0)  ego is behind NPC — ego will rear-end NPC
  side-swipe-left  target_dir = ( 0, -1)  ego is to NPC's right — ego will lane-change left into NPC
  side-swipe-right target_dir = ( 0, +1)  ego is to NPC's left  — ego will lane-change right into NPC

Crash classification reads from vehicles[1] (ego's collision_classification).
Side-swipe edge naming follows highway_env's convention:
  "right" in ego.ego_feature → ego's geometrically-left edge → ego went left → side-swipe-left
  "left"  in ego.ego_feature → ego's geometrically-right edge → ego went right → side-swipe-right
"""

import os

import gymnasium as gym
import numpy as np

VALID_CRASH_TYPES = {"rear-end", "side-swipe-left", "side-swipe-right"}

def _get_env_float(name: str, default: float) -> float:
    """Read runtime reward override from env var, fallback to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


W_SHAPING       = _get_env_float("MASTER_W_SHAPING", 0.03)
R_MATCH         = _get_env_float("MASTER_R_MATCH", 30.0)
R_WRONG         = _get_env_float("MASTER_R_WRONG", 0.0)     # SSL/SSR wrong crash penalty
R_WRONG_RE      = _get_env_float("MASTER_R_WRONG_RE", -10.0)  # RE wrong crash penalty
R_NO_CRASH_RE   = _get_env_float("MASTER_R_NO_CRASH_RE", -0.5)  # RE: avoid collapse to no-crash policy
PROXIMITY_SCALE = 10.0
REAR_END_LAT_TOL = 2.0   # meters, suppress shaping when ego is laterally offset
EPS             = 1e-6

TARGET_DIRS: dict[str, np.ndarray] = {
    "rear-end":         np.array([-1.0,  0.0]),  # ego behind NPC
    "side-swipe-left":  np.array([ 0.0, -1.0]),  # ego to NPC's right
    "side-swipe-right": np.array([ 0.0, +1.0]),  # ego to NPC's left
}


def _compute_mtv_local(env) -> tuple[np.ndarray, float]:
    """NPC→ego vector rotated into NPC's local frame (x=forward, y=left)."""
    vehicles = env.unwrapped.road.vehicles
    if len(vehicles) < 2:
        return np.zeros(2), 0.0
    npc = vehicles[0]
    ego = vehicles[1]

    mtv_world = np.array(ego.position) - np.array(npc.position)
    d = float(np.linalg.norm(mtv_world))

    theta = npc.heading
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    mtv_local = np.array([
         cos_t * mtv_world[0] + sin_t * mtv_world[1],
        -sin_t * mtv_world[0] + cos_t * mtv_world[1],
    ])
    return mtv_local, d


def _classify_from_ego(ego_vehicle) -> str | None:
    """Read crash type from the ego's collision_classification (vehicles[1])."""
    cl = getattr(ego_vehicle, "collision_classification", None)
    if cl is None:
        return None
    c_type = cl.collision_type
    if c_type == "side-swipe":
        if "right" in cl.ego_feature:
            return "side-swipe-left"
        elif "left" in cl.ego_feature:
            return "side-swipe-right"
        return None
    return c_type


class MasterRewardWrapper(gym.Wrapper):

    def __init__(self, env, target_crash_type: str = "rear-end"):
        assert target_crash_type in VALID_CRASH_TYPES, \
            f"Invalid target_crash_type '{target_crash_type}'. Must be one of {VALID_CRASH_TYPES}"
        super().__init__(env)
        self.target_crash_type = target_crash_type
        self._target_dir = TARGET_DIRS[target_crash_type]

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        mtv_local, d = _compute_mtv_local(self.env)
        unit_mtv = mtv_local / max(d, EPS)
        alignment = float(np.dot(unit_mtv, self._target_dir))
        proximity = 1.0 / (1.0 + d / PROXIMITY_SCALE)
        # Rear-end is highly sensitive to lateral offset. Without gating, the policy can
        # collect shaping while setting up side-swipes, then pay only a small terminal penalty.
        if self.target_crash_type == "rear-end":
            lat_dist = abs(float(mtv_local[1]))
            lateral_gate = max(0.0, 1.0 - lat_dist / REAR_END_LAT_TOL)
            dense_reward = W_SHAPING * alignment * proximity * lateral_gate
            info["master_rear_end_lateral_gate"] = float(lateral_gate)
        else:
            dense_reward = W_SHAPING * alignment * proximity

        terminal_bonus = 0.0
        terminal_outcome = "no_crash"
        if (terminated or truncated) and info.get("crashed", False):
            vehicles = self.env.unwrapped.road.vehicles
            if len(vehicles) >= 2:
                crash_type = _classify_from_ego(vehicles[1])
                if crash_type is not None:
                    info["master_crash_type"] = crash_type
                    if crash_type == self.target_crash_type:
                        terminal_bonus = R_MATCH
                        terminal_outcome = "target"
                    elif self.target_crash_type == "rear-end":
                        terminal_bonus = R_WRONG_RE
                        terminal_outcome = "wrong_type"
                        info["master_wrong_crash_type"] = crash_type
                    else:
                        terminal_bonus = -R_WRONG  # = 0.0 for SSL/SSR
                        terminal_outcome = "wrong_type"
                        info["master_wrong_crash_type"] = crash_type
                else:
                    terminal_outcome = "crashed_unknown"
        elif (terminated or truncated) and self.target_crash_type == "rear-end":
            # Rear-end training can collapse into "always avoid collision";
            # assign small terminal cost to no-crash episodes.
            terminal_bonus = R_NO_CRASH_RE

        info["master_terminal_outcome"] = terminal_outcome
        info["master_terminal_bonus"] = terminal_bonus
        info["master_mtv_local_x"]    = float(mtv_local[0])
        info["master_mtv_local_y"]    = float(mtv_local[1])
        info["master_mtv_dist"]       = float(d)
        info["master_alignment"]      = alignment
        info["master_proximity"]      = float(proximity)
        info["master_dense_reward"]   = float(dense_reward)

        return obs, dense_reward + terminal_bonus, terminated, truncated, info
