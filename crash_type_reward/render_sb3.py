"""
Visual rendering of trained SB3 DQN crash-type specialists.

render_mode=human     → opens a pygame window (Ctrl+C to stop)
render_mode=rgb_array → saves per-spawn-config MP4s to --output_dir

Usage:
    # Live view, rear-end model, one specific spawn config:
    uv run python crash_type_reward/render_sb3.py \
        --model new_results/sb3/dqn_rear-end_750k.zip \
        --spawn_config behind_center \
        --episodes 5

    # Live view, all spawn configs (2 eps each):
    uv run python crash_type_reward/render_sb3.py \
        --model new_results/sb3/dqn_side-swipe-left_750k.zip \
        --episodes 2

    # Record all spawn configs → MP4s:
    uv run python crash_type_reward/render_sb3.py \
        --model new_results/sb3/dqn_rear-end_750k.zip \
        --render_mode rgb_array \
        --episodes 3 \
        --output_dir new_results/sb3/videos/
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from dqn_crasher.utils.config import load_pkg_yaml
from sat_reward_wrapper import SATRewardWrapper, VALID_CRASH_TYPES
from train_sb3 import SingleAgentNPCWrapper
from eval_utils import ALL_SPAWN_CONFIGS

_TARGET_RE = re.compile(r"dqn_(?P<target>[a-z\-]+?)_\d+k")


def infer_target(path: str) -> str:
    m = _TARGET_RE.search(os.path.basename(path))
    if not m:
        raise SystemExit(
            f"Cannot infer target from '{os.path.basename(path)}'. "
            f"Expected pattern dqn_<target>_<N>k.zip"
        )
    return m.group("target")


def build_render_env(target_crash_type, spawn_config, render_mode, frame_stack=5):
    gym_config = load_pkg_yaml("configs/env/crash_type_reward.yaml")
    gym_config["observation"]["observation_config"]["frame_stack"] = frame_stack
    if spawn_config != "random":
        gym_config["spawn_configs"] = [spawn_config]

    env = gym.make("crash-v0", config=gym_config, render_mode=render_mode)
    env = SATRewardWrapper(env, target_crash_type=target_crash_type)
    env = SingleAgentNPCWrapper(env, frame_stack=frame_stack)
    env = Monitor(env)
    return env


def run_episodes(model, env, n_episodes, render_mode, seed=0):
    """Run episodes and return per-episode frame lists + stats."""
    all_episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
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

        # Capture terminal frame
        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        crashed = info.get("crashed", False)
        crash_type = info.get("sat_crash_type") if crashed else None
        all_episodes.append({
            "frames": frames,
            "crashed": crashed,
            "crash_type": crash_type or "none",
            "reward": ep_reward,
            "length": ep_len,
        })
        print(
            f"    ep {ep + 1:2d}: crashed={str(crashed):<5}  "
            f"type={crash_type or 'none':<22}  "
            f"reward={ep_reward:6.2f}  len={ep_len}"
        )

    return all_episodes


def save_video(frames, out_path, fps=15):
    if not frames:
        print(f"  [warn] No frames collected — skipping {out_path}")
        return
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
        clip.write_videofile(out_path, logger=None)
    except Exception as e:
        # moviepy 2.x moved some things; try top-level import
        try:
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
            clip.write_videofile(out_path, logger=None)
        except Exception:
            import imageio
            imageio.mimsave(out_path, [np.array(f) for f in frames], fps=fps)
    print(f"  Saved → {out_path}  ({len(frames)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Render trained SB3 DQN crash-type models")
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument(
        "--spawn_config",
        default="all",
        help=(
            f"Which spawn config to use, or 'all' to iterate all 8. "
            f"Valid: {ALL_SPAWN_CONFIGS} (default: all)"
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=2,
        help="Episodes per spawn config (default: 2)",
    )
    parser.add_argument(
        "--render_mode",
        default="human",
        choices=["human", "rgb_array"],
        help="human = live pygame window; rgb_array = save MP4s (default: human)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for MP4 output (required with render_mode=rgb_array)",
    )
    parser.add_argument("--fps", type=int, default=15, help="FPS for saved videos (default: 15)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stack", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    if args.render_mode == "rgb_array" and args.output_dir is None:
        raise SystemExit("--output_dir is required when using render_mode=rgb_array")

    target = infer_target(args.model)
    model_stem = os.path.splitext(os.path.basename(args.model))[0]

    print(f"Model      : {os.path.basename(args.model)}")
    print(f"Target     : {target}")
    print(f"Render mode: {args.render_mode}")

    if args.spawn_config == "all":
        spawn_configs = ALL_SPAWN_CONFIGS
    elif args.spawn_config == "random":
        spawn_configs = ["random"]
    elif args.spawn_config in ALL_SPAWN_CONFIGS:
        spawn_configs = [args.spawn_config]
    else:
        raise SystemExit(
            f"Unknown spawn_config '{args.spawn_config}'. "
            f"Valid: {ALL_SPAWN_CONFIGS}, 'all', or 'random'"
        )

    model = DQN.load(args.model)

    # per-model subdir so videos from different models don't collide
    out_dir = os.path.join(args.output_dir, target) if args.output_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for cfg in spawn_configs:
        print(f"\n[{cfg}]")
        env = build_render_env(target, cfg, args.render_mode, frame_stack=args.frame_stack)
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
