"""
Per-step trajectory diagnostic for a single (model, spawn_config) cell.

Loads a saved SB3 checkpoint, runs N episodes with the env forced to one
spawn_config, and dumps every step to stdout + JSONL. Useful for
understanding WHY a model fails on a specific cell (e.g., rear-end model
at adjacent_left → 0% target-hit).

Per step logs:
  ego (victim) pos (x, y, lane), vel
  npc (controlled) pos, vel, lane
  action taken by NPC (LANE_LEFT=0, IDLE=1, LANE_RIGHT=2, FASTER=3, SLOWER=4)
  reward (from SAT wrapper)
  terminal info: crashed, sat_crash_type, collision_classification
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from stable_baselines3 import DQN

from eval_utils import build_env

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


def _vehicle_state(v):
    return {
        "x": float(v.position[0]),
        "y": float(v.position[1]),
        "vx": float(v.velocity[0]),
        "vy": float(v.velocity[1]),
        "speed": float(v.speed),
        "lane": list(v.lane_index[-1:]) if v.lane_index else None,
    }


def run_cell(model_path, target, spawn_config, n_episodes=5, seed=0, frame_stack=5):
    env = build_env(target, frame_stack=frame_stack, spawn_configs=[spawn_config])
    model = DQN.load(model_path)

    episodes = []
    # Access the underlying crash env for vehicle inspection
    def _unwrap(e):
        while hasattr(e, "env"):
            e = e.env
        return e

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        raw_env = _unwrap(env)

        def _pick_pair():
            # Prefer road.vehicles (always has both), fall back to env.vehicles
            vs = getattr(raw_env.road, "vehicles", None) if hasattr(raw_env, "road") else None
            if vs is None or len(vs) < 2:
                vs = getattr(raw_env, "vehicles", None) or []
            if len(vs) >= 2:
                return vs[0], vs[1]
            return (vs[0] if vs else getattr(raw_env, "vehicle", None)), None

        npc, victim = _pick_pair()

        steps = []
        terminated = truncated = False
        step_idx = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            # Re-pick each step in case vehicle list changed
            npc, victim = _pick_pair()
            npc_now = _vehicle_state(npc) if npc is not None else None
            vic_now = _vehicle_state(victim) if victim is not None else None

            obs, reward, terminated, truncated, info = env.step(action)
            dx = dy = None
            if npc_now is not None and vic_now is not None:
                dx = vic_now["x"] - npc_now["x"]
                dy = vic_now["y"] - npc_now["y"]
            steps.append({
                "step": step_idx,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "reward": float(reward),
                "npc": npc_now,
                "victim": vic_now,
                "dx": dx,
                "dy": dy,
            })
            step_idx += 1

        term_info = {
            "crashed": bool(info.get("crashed", False)),
            "sat_crash_type": info.get("sat_crash_type", None),
            "sat_terminal_bonus": info.get("sat_terminal_bonus", 0.0),
            "spawn_config": info.get("spawn_config", None),
            "num_steps": step_idx,
        }
        # Pull collision_classification from the NPC for mismatch diagnosis
        cl = getattr(npc, "collision_classification", None)
        if cl is not None:
            term_info["collision_type"] = getattr(cl, "collision_type", None)
            term_info["ego_feature"] = getattr(cl, "ego_feature", None)
            term_info["other_feature"] = getattr(cl, "other_feature", None)

        episodes.append({"ep": ep_idx, "terminal": term_info, "steps": steps})

    env.close()
    return episodes


def summarize(episodes, target):
    from collections import Counter
    n = len(episodes)
    crashed = sum(1 for e in episodes if e["terminal"]["crashed"])
    target_hits = sum(1 for e in episodes if e["terminal"]["sat_crash_type"] == target)
    sat_types = Counter(e["terminal"]["sat_crash_type"] or "uncrashed" for e in episodes)
    coll_types = Counter(e["terminal"].get("collision_type", "none") for e in episodes)
    lens = [e["terminal"]["num_steps"] for e in episodes]
    first_actions = Counter(e["steps"][0]["action_name"] for e in episodes if e["steps"])

    # Action distribution across all steps
    all_actions = Counter()
    for e in episodes:
        for s in e["steps"]:
            all_actions[s["action_name"]] += 1

    print(f"\n  episodes:         {n}")
    print(f"  any_crash:        {crashed}/{n} = {crashed/n:.0%}")
    print(f"  target_hit:       {target_hits}/{n} = {target_hits/n:.0%}  (target: {target})")
    print(f"  episode_lens:     min={min(lens)} max={max(lens)} mean={np.mean(lens):.1f}")
    print(f"  sat_crash_type:   {dict(sat_types)}")
    print(f"  collision_type:   {dict(coll_types)}")
    print(f"  first-step action:{dict(first_actions)}")
    print(f"  action dist (all):{dict(all_actions)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--spawn_config", required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_jsonl", default=None,
                   help="If given, dump full per-step trajectories to this JSONL file")
    p.add_argument("--verbose", action="store_true",
                   help="Print every step of every episode (verbose)")
    args = p.parse_args()

    print(f"\n=== {os.path.basename(args.model)} @ {args.spawn_config} (target={args.target}) ===")
    episodes = run_cell(
        args.model, args.target, args.spawn_config,
        n_episodes=args.n_episodes, seed=args.seed,
    )

    if args.verbose:
        for e in episodes:
            print(f"\n  --- ep {e['ep']} ({e['terminal']['num_steps']} steps, crashed={e['terminal']['crashed']}, sat_type={e['terminal']['sat_crash_type']}, coll_type={e['terminal'].get('collision_type')}) ---")
            for s in e["steps"]:
                vic = s["victim"]
                npc = s["npc"]
                npc_str = f"({npc['x']:>7.2f},{npc['y']:>6.2f}) v={npc['speed']:>5.2f}" if npc else "(npc=?)"
                vic_str = f"({vic['x']:>7.2f},{vic['y']:>6.2f}) v={vic['speed']:>5.2f}" if vic else "(vic=?)"
                dx_str = f"{s['dx']:>+6.2f}" if s['dx'] is not None else "   --"
                dy_str = f"{s['dy']:>+6.2f}" if s['dy'] is not None else "   --"
                print(
                    f"    t={s['step']:>2} a={s['action_name']:<10} "
                    f"npc={npc_str}  vic={vic_str}  "
                    f"dx={dx_str} dy={dy_str}  r={s['reward']:>+.2f}"
                )

    summarize(episodes, args.target)

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")
        print(f"\n  full trajectories → {args.output_jsonl}")


if __name__ == "__main__":
    main()
