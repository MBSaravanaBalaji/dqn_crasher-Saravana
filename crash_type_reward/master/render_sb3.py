"""
Visual rendering of trained master NPC DQN models.

Usage:
    # Rear-end with training-matched scenario knobs:
    uv run python crash_type_reward/master/render_sb3.py \
        --model new_results/master/master_rear-end_500k_best.zip \
        --spawn_config forward_left --apply_rear_end_defaults --episodes 5

    # Live view, all spawn configs:
    uv run python crash_type_reward/master/render_sb3.py \
        --model new_results/master/master_side-swipe-left_750k.zip \
        --episodes 2

    # Record MP4s:
    uv run python crash_type_reward/master/render_sb3.py \
        --model new_results/master/master_rear-end_500k_best.zip \
        --spawn_config forward_right --apply_rear_end_defaults \
        --render_mode rgb_array --episodes 3 \
        --output_dir new_results/master/videos/
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import numpy as np
from stable_baselines3 import DQN

from eval_utils import (
    ALL_SPAWN_CONFIGS,
    _EGO_TYPES,
    REAR_END_TRAINING_DEFAULTS,
    build_env,
    get_spawn_configs_for_eval_mode,
)
from train_sb3 import DistanceRandomizerWrapper

_TARGET_RE = re.compile(r"master_(?P<target>[a-z\-]+?)_\d+k")


def infer_target(path: str) -> str:
    m = _TARGET_RE.search(os.path.basename(path))
    if not m:
        raise SystemExit(
            f"Cannot infer target from '{os.path.basename(path)}'. "
            f"Expected pattern master_<target>_<N>k.zip"
        )
    return m.group("target")


def _apply_rear_end_defaults(args) -> None:
    defaults = REAR_END_TRAINING_DEFAULTS
    if not args.no_mobil:
        args.no_mobil = defaults["no_mobil"]
    if args.mean_delta_v is None:
        args.mean_delta_v = defaults["mean_delta_v"]
    if args.target_speeds is None:
        args.target_speeds = defaults["target_speeds"]
    if args.policy_frequency is None:
        args.policy_frequency = defaults["policy_frequency"]
    if args.duration is None:
        args.duration = defaults["duration"]
    if args.distance_range is None:
        args.distance_range = defaults["distance_range"]
    if args.ego_type == "fixed":
        args.ego_type = defaults["ego_type"]


def build_render_env(
    target_crash_type,
    spawn_config,
    render_mode,
    frame_stack=5,
    *,
    mean_distance=None,
    mean_delta_v=None,
    distance_range=None,
    ego_type="fixed",
    no_mobil=False,
    target_speeds=None,
    policy_frequency=None,
    duration=None,
):
    target_speeds_list = (
        [float(x) for x in target_speeds.split(",")] if target_speeds else None
    )
    spawn_configs = None if spawn_config == "random" else [spawn_config]

    env = build_env(
        target_crash_type,
        frame_stack=frame_stack,
        spawn_configs=spawn_configs,
        mean_distance=mean_distance,
        mean_delta_v=mean_delta_v,
        ego_type=ego_type,
        no_mobil=no_mobil,
        target_speeds=target_speeds_list,
        policy_frequency=policy_frequency,
        duration=duration,
        render_mode=render_mode,
    )

    if distance_range is not None:
        parts = distance_range.split(",")
        d_min, d_max = float(parts[0].strip()), float(parts[1].strip())
        env = DistanceRandomizerWrapper(env, d_min=d_min, d_max=d_max)
    return env


def run_episodes(model, env, n_episodes, render_mode, seed=0):
    all_episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        frames = []
        ep_reward = 0.0
        ep_len = 0

        while not (terminated or truncated):
            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif render_mode == "human":
                env.render()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        crashed = info.get("crashed", False)
        crash_type = info.get("master_crash_type") if crashed else None
        outcome = info.get("master_terminal_outcome", "no_crash")
        all_episodes.append({
            "frames": frames,
            "crashed": crashed,
            "crash_type": crash_type or "none",
            "outcome": outcome,
            "reward": ep_reward,
            "length": ep_len,
        })
        print(
            f"    ep {ep + 1:2d}: crashed={str(crashed):<5}  "
            f"outcome={outcome:<14}  "
            f"type={crash_type or 'none':<22}  "
            f"reward={ep_reward:6.2f}  len={ep_len}"
        )

    return all_episodes


def save_video(frames, out_path, fps=15):
    if not frames:
        print(f"  [warn] No frames — skipping {out_path}")
        return
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
        clip.write_videofile(out_path, logger=None)
    except Exception:
        try:
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
            clip.write_videofile(out_path, logger=None)
        except Exception:
            import imageio
            imageio.mimsave(out_path, [np.array(f) for f in frames], fps=fps)
    print(f"  Saved → {out_path}  ({len(frames)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Render trained master NPC DQN models")
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument(
        "--spawn_config", default="compatible",
        help=f"Spawn: name, 'all', 'random', or 'compatible' (target-dependent). "
             f"Valid names: {ALL_SPAWN_CONFIGS}",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--render_mode", default="human", choices=["human", "rgb_array"],
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stack", type=int, default=5)
    parser.add_argument(
        "--apply_rear_end_defaults", action="store_true",
        help="Apply REAR_END_TRAINING_DEFAULTS (no_mobil, delta_v, speeds, etc.).",
    )
    parser.add_argument("--no_mobil", action="store_true")
    parser.add_argument("--mean_distance", type=float, default=None)
    parser.add_argument("--mean_delta_v", type=float, default=None)
    parser.add_argument("--distance_range", default=None, help="e.g. '15,30'")
    parser.add_argument(
        "--ego_type", default="fixed",
        choices=list(_EGO_TYPES),
    )
    parser.add_argument("--target_speeds", default=None)
    parser.add_argument("--policy_frequency", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    if args.render_mode == "rgb_array" and args.output_dir is None:
        raise SystemExit("--output_dir is required with render_mode=rgb_array")

    if args.apply_rear_end_defaults:
        _apply_rear_end_defaults(args)

    target = infer_target(args.model)
    model_stem = os.path.splitext(os.path.basename(args.model))[0]

    print(f"Model      : {os.path.basename(args.model)}")
    print(f"Target     : {target}")
    print(f"Render mode: {args.render_mode}")
    if args.no_mobil or args.ego_type == "no_mobil":
        print("Ego        : NoMobilEgoVehicle")
    if args.mean_delta_v is not None:
        print(f"Mean delta-v: {args.mean_delta_v}")
    if args.distance_range:
        print(f"Distance range: {args.distance_range}")

    if args.spawn_config == "all":
        spawn_configs = ALL_SPAWN_CONFIGS
    elif args.spawn_config == "compatible":
        spawn_configs = get_spawn_configs_for_eval_mode(target, "compatible")
    elif args.spawn_config == "random":
        spawn_configs = ["random"]
    elif args.spawn_config in ALL_SPAWN_CONFIGS:
        spawn_configs = [args.spawn_config]
    else:
        raise SystemExit(
            f"Unknown spawn_config '{args.spawn_config}'. "
            f"Valid: {ALL_SPAWN_CONFIGS}, 'all', 'compatible', or 'random'"
        )

    model = DQN.load(args.model)

    out_dir = os.path.join(args.output_dir, target) if args.output_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for cfg in spawn_configs:
        print(f"\n[{cfg}]")
        env = build_render_env(
            target,
            cfg,
            args.render_mode,
            frame_stack=args.frame_stack,
            mean_distance=args.mean_distance,
            mean_delta_v=args.mean_delta_v,
            distance_range=args.distance_range,
            ego_type=args.ego_type,
            no_mobil=args.no_mobil,
            target_speeds=args.target_speeds,
            policy_frequency=args.policy_frequency,
            duration=args.duration,
        )
        try:
            episodes_data = run_episodes(
                model, env, args.episodes, args.render_mode, seed=args.seed
            )
        finally:
            env.close()

        if args.render_mode == "rgb_array" and out_dir:
            all_frames = []
            for ep_data in episodes_data:
                all_frames.extend(ep_data["frames"])
            out_path = os.path.join(out_dir, f"{model_stem}_{cfg}.mp4")
            save_video(all_frames, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
