"""
Gym wrapper: per-step MTV shaping reward + terminal crash-type reward.

Dense shaping: alignment of ego→NPC direction (in ego local frame) with the
target approach geometry, weighted by proximity. Gives a non-zero gradient every step.

Terminal: +R_MATCH if crash matches target_crash_type, -R_WRONG for wrong-type crash.

Convention:
  vehicles[0] = NPC adversary (controlled, created first)
  vehicles[1] = ego (IDM)

MTV direction is ego→NPC in ego's local frame (x=forward, y=left):
  rear-end         target_dir = (-1,  0)  NPC behind ego, approaches from rear
  rear-ended       target_dir = (+1,  0)  NPC ahead of ego, ego drives into NPC's rear
  side-swipe-left  target_dir = ( 0, +1)  NPC approaching from ego's left  (higher y)
  side-swipe-right target_dir = ( 0, -1)  NPC approaching from ego's right (lower y)

Crash classification follows the NPC's collision_classification ego_feature:
  "side-swipe" + NPC ego_feature "right" → NPC to ego's left  → "side-swipe-left"
  "side-swipe" + NPC ego_feature "left"  → NPC to ego's right → "side-swipe-right"

Sign confirmed by smoke test on data/NPC_*.jsonl:
  In highway_env y increases to the left, so NPC on ego's left → ego→NPC y > 0.
  side-swipe-left mean y = +2.27, side-swipe-right mean y = -1.38.
"""

import gymnasium as gym
import numpy as np

VALID_CRASH_TYPES = {"rear-end", "rear-ended", "side-swipe-left", "side-swipe-right"}

# Tuning knobs — change here, nowhere else
W_SHAPING       = 0.1    # scale of per-step shaping term; << R_MATCH
R_MATCH         = 10.0   # terminal bonus for target-type crash
R_WRONG         = 0.0    # terminal penalty for wrong-type crash; 0 = no discouragement (side-swipe-left was avoidance-collapsing at 2.0)
TIME_PENALTY    = 0.0    # per-step penalty; set to -0.01 if policy dithers
PROXIMITY_SCALE = 10.0   # distance (m) at which proximity ≈ 0.5
EPS             = 1e-6

TARGET_DIRS: dict[str, np.ndarray] = {
    "rear-end":          np.array([-1.0,  0.0]),   # NPC behind ego, approaches from rear
    "rear-ended":        np.array([+1.0,  0.0]),   # NPC ahead of ego, ego drives into NPC's rear
    "side-swipe-left":   np.array([ 0.0, +1.0]),   # NPC to ego's left  (higher y)
    "side-swipe-right":  np.array([ 0.0, -1.0]),   # NPC to ego's right (lower y)
}


def _compute_mtv_local(env) -> tuple[np.ndarray, float]:
    """
    Return (mtv_local, distance) where mtv_local is the ego→NPC vector
    rotated into the ego's local frame (x=forward, y=left).

    Uses vehicle centroids — valid for disjoint polygons (99% of steps).
    Returns (zeros, 0.0) if vehicles are not available.
    """
    vehicles = env.unwrapped.road.vehicles
    if len(vehicles) < 2:
        return np.zeros(2), 0.0
    npc    = vehicles[0]
    ego = vehicles[1]

    mtv_world = np.array(npc.position) - np.array(ego.position)
    d = float(np.linalg.norm(mtv_world))

    theta = ego.heading
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # R(-theta): rotate world coords into ego local frame
    mtv_local = np.array([
         cos_t * mtv_world[0] + sin_t * mtv_world[1],
        -sin_t * mtv_world[0] + cos_t * mtv_world[1],
    ])
    return mtv_local, d


def _classify_from_vehicle(npc_vehicle):
    """Read crash type from the NPC's internal collision_classification."""
    cl = getattr(npc_vehicle, "collision_classification", None)
    if cl is None:
        return None
    c_type = cl.collision_type
    if c_type == "side-swipe":
        if "right" in cl.ego_feature:
            return "side-swipe-left"    # NPC's right hit ego's left
        elif "left" in cl.ego_feature:
            return "side-swipe-right"   # NPC's left hit ego's right
        return None
    return c_type


class SATRewardWrapper(gym.Wrapper):

    def __init__(self, env, target_crash_type="rear-end"):
        assert target_crash_type in VALID_CRASH_TYPES, \
            f"Invalid target_crash_type '{target_crash_type}'. Must be one of {VALID_CRASH_TYPES}"
        super().__init__(env)
        self.target_crash_type = target_crash_type
        self._target_dir = TARGET_DIRS[target_crash_type]

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Dense MTV shaping (every step)
        mtv_local, d = _compute_mtv_local(self.env)
        unit_mtv = mtv_local / max(d, EPS)
        alignment = float(np.dot(unit_mtv, self._target_dir))
        proximity = 1.0 / (1.0 + d / PROXIMITY_SCALE)
        dense_reward = W_SHAPING * alignment * proximity

        # Terminal reward
        terminal_bonus = 0.0
        if (terminated or truncated) and info.get("crashed", False):
            vehicles = self.env.unwrapped.road.vehicles
            if len(vehicles) >= 1:
                crash_type = _classify_from_vehicle(vehicles[0])
                if crash_type is not None:
                    info["sat_crash_type"] = crash_type
                    if crash_type == self.target_crash_type:
                        terminal_bonus = R_MATCH
                    else:
                        terminal_bonus = -R_WRONG

        info["sat_terminal_bonus"] = terminal_bonus
        info["sat_mtv_local_x"]   = float(mtv_local[0])
        info["sat_mtv_local_y"]   = float(mtv_local[1])
        info["sat_mtv_dist"]      = float(d)
        info["sat_alignment"]     = alignment
        info["sat_proximity"]     = float(proximity)
        info["sat_dense_reward"]  = float(dense_reward)

        total_reward = dense_reward + TIME_PENALTY + terminal_bonus
        return obs, total_reward, terminated, truncated, info
