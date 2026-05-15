"""
Stress-test evaluator for master NPC DQN checkpoints.

Usage:
    # Rear-end with training-matched knobs and compatible spawns only:
    uv run python crash_type_reward/master/eval_sb3.py \
        --models new_results/master/master_rear-end_500k_best.zip \
        --eval_mode compatible --apply_rear_end_defaults \
        --episodes_per_cell 200

    uv run python crash_type_reward/master/eval_sb3.py \
        --models new_results/master/master_side-swipe-left_750k.zip \
        --episodes_per_cell 200

    # Full 8-spawn grid (generalization):
    uv run python crash_type_reward/master/eval_sb3.py \
        --models new_results/master/master_rear-end_500k_best.zip \
        --eval_mode all --no_mobil --mean_delta_v -12 --episodes_per_cell 200

    # Stress grid over distances and delta-v:
    uv run python crash_type_reward/master/eval_sb3.py \
        --models new_results/master/master_rear-end_500k_best.zip \
        --stress --apply_rear_end_defaults --episodes_per_cell 50
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from stable_baselines3 import DQN

from eval_utils import (
    ALL_SPAWN_CONFIGS,
    COMPATIBLE_CONFIGS_BY_TARGET,
    EVAL_MODES,
    INFEASIBLE_CELLS_BY_TARGET,
    _EGO_TYPES,
    REAR_END_TRAINING_DEFAULTS,
    CellResult,
    format_crash_type_breakdown,
    get_spawn_configs_for_eval_mode,
    run_stress_eval,
    summary_metrics,
)

_TARGET_RE = re.compile(r"master_(?P<target>[a-z\-]+?)_\d+k")


def infer_target_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = _TARGET_RE.search(name)
    if not m:
        raise SystemExit(
            f"Cannot infer target from '{name}'. "
            f"Expected pattern 'master_<target>_<N>k.zip'."
        )
    return m.group("target")


def _apply_rear_end_defaults(args) -> None:
    """Fill unset rear-end eval knobs from REAR_END_TRAINING_DEFAULTS."""
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
    if args.ego_type == "fixed":
        args.ego_type = defaults["ego_type"]


def print_per_config_table(results_by_model: dict[str, list[CellResult]]) -> None:
    print()
    print("=" * 110)
    print("Per-spawn_config target-hit rate (compatible configs marked with *)")
    print("=" * 110)
    header = f"{'model':<45} {'target':<17} " + " ".join(f"{c:>10}" for c in ALL_SPAWN_CONFIGS)
    print(header)
    print("-" * len(header))

    for model_path, cells in results_by_model.items():
        if not cells:
            continue
        target = cells[0].target_crash_type
        compat = set(COMPATIBLE_CONFIGS_BY_TARGET.get(target, []))
        infeasible = INFEASIBLE_CELLS_BY_TARGET.get(target, set())
        by_config = {c.spawn_config: c for c in cells}
        row = f"{os.path.basename(model_path):<45} {target:<17} "
        for cfg in ALL_SPAWN_CONFIGS:
            r = by_config.get(cfg)
            if r is None:
                cell_str = "       -  "
            else:
                if cfg in compat:
                    mark = "*"
                elif cfg in infeasible:
                    mark = "x"
                else:
                    mark = " "
                cell_str = f"{mark}{r.target_hit_rate * 100:>6.1f}%  "
            row += cell_str
        print(row)

    print()
    print("* = compatible spawn config for this target (NPC starts in favourable position)")
    print("x = structurally infeasible (NPC on wrong side or adjacent with dx=0)")
    print()


def print_crash_breakdown(results_by_model: dict[str, list[CellResult]]) -> None:
    print("Crash-type breakdown per cell (when any crash occurred):")
    for model_path, cells in results_by_model.items():
        print(f"  {os.path.basename(model_path)}:")
        for r in cells:
            if r.n_crashed == 0:
                continue
            print(
                f"    {r.spawn_config:<16} target={r.target_hit_rate:.1%}  "
                f"wrong={r.wrong_type_crash_rate:.1%}  types: {format_crash_type_breakdown(r)}"
            )
    print()


def print_summary_metrics(results_by_model: dict[str, list[CellResult]]) -> None:
    print("Summary metrics per model:")
    print(
        f"  {'model':<45} {'mean_compat':>12} {'min_compat':>12} "
        f"{'mean_feas':>12} {'mean_all':>12} {'any_crash':>12} {'wrong_cr':>12}"
    )
    for model_path, cells in results_by_model.items():
        if not cells:
            continue
        target = cells[0].target_crash_type
        m = summary_metrics(cells, target)
        print(
            f"  {os.path.basename(model_path):<45} "
            f"{m['eval/mean_compatible_target_hit']:>12.3f} "
            f"{m['eval/min_compatible_target_hit']:>12.3f} "
            f"{m['eval/mean_feasible_target_hit']:>12.3f} "
            f"{m['eval/mean_all_configs_target_hit']:>12.3f} "
            f"{m['eval/mean_any_crash_rate']:>12.3f} "
            f"{m['eval/mean_wrong_type_crash_rate']:>12.3f}"
        )
    print()


def write_csv(results_by_model: dict[str, list[CellResult]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "model_path", "target", "spawn_config",
        "mean_distance", "mean_delta_v", "use_spawn_distribution",
        "n_episodes", "n_crashed", "n_target_hit", "n_crashed_unknown",
        "target_hit_rate", "any_crash_rate", "wrong_type_crash_rate",
        "mean_episode_length", "observed_crash_types",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cells in results_by_model.values():
            for r in cells:
                writer.writerow(r.as_row())
    print(f"Per-cell results written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", required=True,
        help="Comma-separated list of .zip model paths. Target inferred from filename.",
    )
    parser.add_argument("--episodes_per_cell", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stack", type=int, default=5)
    parser.add_argument(
        "--eval_mode", default="compatible", choices=list(EVAL_MODES),
        help="Spawn grid: compatible (default for RE), feasible (excl. infeasible), all (8 spawns).",
    )
    parser.add_argument(
        "--apply_rear_end_defaults", action="store_true",
        help="Apply REAR_END_TRAINING_DEFAULTS for no_mobil, delta_v, speeds, policy_frequency, etc.",
    )
    parser.add_argument(
        "--stress", action="store_true",
        help="Sweep mean_distance and mean_delta_v per cell.",
    )
    parser.add_argument("--distance_grid", default="10,20,30,40")
    parser.add_argument("--delta_v_grid", default="-12,-5,0")
    parser.add_argument(
        "--mean_distance", type=float, default=None,
        help="Override spawn mean_distance for non-stress eval.",
    )
    parser.add_argument(
        "--mean_delta_v", type=float, default=None,
        help="NPC speed offset vs ego (negative → ego closes from behind).",
    )
    parser.add_argument(
        "--initial_lane_id", type=int, default=None,
        help="Force ego to start in this lane (e.g. 2 for SSR to replicate training conditions).",
    )
    parser.add_argument(
        "--ego_type", default="fixed",
        choices=list(_EGO_TYPES),
        help="Ego vehicle type for eval.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Defaults to new_results/master/eval_<eval_mode>_<ego_type>.csv",
    )
    parser.add_argument(
        "--no_mobil", action="store_true",
        help="Disable MOBIL lane-change for ego during eval.",
    )
    parser.add_argument(
        "--target_speeds", default=None,
        help="Comma-separated NPC target speeds, e.g. '5,10,15,20,25,30,34'.",
    )
    parser.add_argument(
        "--policy_frequency", type=int, default=None,
        help="Override policy_frequency (Hz). Use 5 for rear-end.",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Override episode duration in steps.",
    )
    args = parser.parse_args()

    if args.apply_rear_end_defaults:
        _apply_rear_end_defaults(args)

    output = args.output or os.path.join(
        os.path.dirname(__file__), "..", "..", "new_results", "master",
        f"eval_{args.eval_mode}_{args.ego_type}.csv",
    )
    args.output = output

    model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Model files not found: {missing}")

    distance_grid = (
        [float(x) for x in args.distance_grid.split(",")] if args.stress else None
    )
    delta_v_grid = (
        [float(x) for x in args.delta_v_grid.split(",")] if args.stress else None
    )
    target_speeds_list = (
        [float(x) for x in args.target_speeds.split(",")]
        if args.target_speeds else None
    )

    results_by_model: dict[str, list[CellResult]] = {}

    for model_path in model_paths:
        target = infer_target_from_filename(model_path)
        spawn_configs = get_spawn_configs_for_eval_mode(target, args.eval_mode)
        n_cells = len(spawn_configs)
        if args.stress:
            n_cells *= len(distance_grid) * len(delta_v_grid)
        total_eps = n_cells * args.episodes_per_cell
        print(
            f"\n--- {os.path.basename(model_path)} (target: {target}, eval_mode: {args.eval_mode}) ---"
        )
        print(f"  Spawns: {spawn_configs}")
        print(f"  Cells: {n_cells} x {args.episodes_per_cell} eps = {total_eps} episodes")

        model = DQN.load(model_path)

        def _on_cell_done(r: CellResult) -> None:
            tag = f"d={r.mean_distance} dv={r.mean_delta_v}" if args.stress else ""
            breakdown = format_crash_type_breakdown(r) if r.n_crashed else ""
            print(
                f"  [{os.path.basename(r.model_path):<40}] "
                f"{r.spawn_config:<16} {tag:<18} "
                f"target_hit={r.target_hit_rate:.1%}  any_crash={r.any_crash_rate:.1%}  "
                f"wrong={r.wrong_type_crash_rate:.1%}  "
                f"mean_len={r.mean_episode_length:.1f}"
                + (f"  [{breakdown}]" if breakdown else "")
            )

        cells = run_stress_eval(
            model=model,
            target_crash_type=target,
            model_path=model_path,
            episodes_per_cell=args.episodes_per_cell,
            spawn_configs=spawn_configs,
            distance_grid=distance_grid,
            delta_v_grid=delta_v_grid,
            distance_override=args.mean_distance if not args.stress else None,
            initial_lane_id=args.initial_lane_id,
            seed=args.seed,
            frame_stack=args.frame_stack,
            ego_type=args.ego_type,
            no_mobil=args.no_mobil,
            target_speeds=target_speeds_list,
            policy_frequency=args.policy_frequency,
            duration=args.duration,
            on_cell_done=_on_cell_done,
        )
        results_by_model[model_path] = cells

    if not args.stress and args.eval_mode == "all":
        print_per_config_table(results_by_model)
    print_crash_breakdown(results_by_model)
    print_summary_metrics(results_by_model)
    write_csv(results_by_model, output)


if __name__ == "__main__":
    main()
