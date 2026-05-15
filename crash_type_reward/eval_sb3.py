"""
Stress-test evaluator for SB3 SAT-reward DQN checkpoints.

For each (model, spawn_config) cell, runs N episodes with the env forced
to that single spawn_config and reports target_hit_rate / any_crash_rate /
mean_episode_length. With --stress, also sweeps mean_distance and
mean_delta_v.

Usage:
    python crash_type_reward/eval_sb3.py \
        --models new_results/sb3/dqn_rear-end_200k.zip,...\
        --episodes_per_cell 200 --seed 0

    # stress grid:
    python crash_type_reward/eval_sb3.py --models <same> --stress --episodes_per_cell 50
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stable_baselines3 import DQN

from eval_utils import (
    ALL_SPAWN_CONFIGS,
    COMPATIBLE_CONFIGS_BY_TARGET,
    INFEASIBLE_CELLS_BY_TARGET,
    CellResult,
    run_stress_eval,
    summary_metrics,
)


_TARGET_RE = re.compile(r"dqn_(?P<target>[a-z\-]+?)_\d+k")


def infer_target_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = _TARGET_RE.search(name)
    if not m:
        raise SystemExit(
            f"Cannot infer target crash type from '{name}'. Expected pattern "
            f"'dqn_<target>_<N>k.zip'. Rename the file or extend the regex."
        )
    return m.group("target")


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
    print("* = spawn_config compatible with the target crash type per the YAML geometry notes")
    print("x = structurally infeasible (mirrored geometry or dx=0 adjacent artifact); excluded from feasible aggregate")
    print()


def print_summary_metrics(results_by_model: dict[str, list[CellResult]]) -> None:
    print("Summary metrics per model:")
    print(f"  {'model':<45} {'mean_compat':>12} {'min_compat':>12} {'mean_feas':>12} {'mean_all':>12} {'any_crash':>12}")
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
            f"{m['eval/mean_any_crash_rate']:>12.3f}"
        )
    print()


def write_csv(results_by_model: dict[str, list[CellResult]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "model_path", "target", "spawn_config",
        "mean_distance", "mean_delta_v", "use_spawn_distribution",
        "n_episodes", "n_crashed", "n_target_hit", "n_crashed_unknown",
        "target_hit_rate", "any_crash_rate", "mean_episode_length",
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
        "--models",
        required=True,
        help="Comma-separated list of .zip model paths. Target crash type is "
             "inferred from filename (dqn_<target>_<N>k.zip).",
    )
    parser.add_argument("--episodes_per_cell", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stack", type=int, default=5)
    parser.add_argument(
        "--stress", action="store_true",
        help="Also sweep mean_distance and mean_delta_v per cell (bigger grid).",
    )
    parser.add_argument(
        "--distance_grid",
        default="10,20,30,40",
        help="Comma list of mean_distance values for --stress",
    )
    parser.add_argument(
        "--delta_v_grid",
        default="-5,0,5",
        help="Comma list of mean_delta_v values for --stress",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "new_results", "sb3", "eval",
            "eval_per_config.csv",
        ),
    )
    args = parser.parse_args()

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

    results_by_model: dict[str, list[CellResult]] = {}

    total_cells = len(model_paths) * len(ALL_SPAWN_CONFIGS)
    if args.stress:
        total_cells *= len(distance_grid) * len(delta_v_grid)
    print(f"Running {total_cells} cells x {args.episodes_per_cell} eps = "
          f"{total_cells * args.episodes_per_cell} episodes")

    def _on_cell_done(r: CellResult) -> None:
        tag = f"d={r.mean_distance} dv={r.mean_delta_v}" if args.stress else ""
        print(
            f"  [{os.path.basename(r.model_path):<40}] "
            f"{r.spawn_config:<16} {tag:<18} "
            f"target_hit={r.target_hit_rate:.1%}  any_crash={r.any_crash_rate:.1%}  "
            f"mean_len={r.mean_episode_length:.1f}"
        )

    for model_path in model_paths:
        target = infer_target_from_filename(model_path)
        print(f"\n--- {os.path.basename(model_path)} (target: {target}) ---")
        model = DQN.load(model_path)

        cells = run_stress_eval(
            model=model,
            target_crash_type=target,
            model_path=model_path,
            episodes_per_cell=args.episodes_per_cell,
            spawn_configs=ALL_SPAWN_CONFIGS,
            distance_grid=distance_grid,
            delta_v_grid=delta_v_grid,
            seed=args.seed,
            frame_stack=args.frame_stack,
            on_cell_done=_on_cell_done,
        )
        results_by_model[model_path] = cells

    if not args.stress:
        print_per_config_table(results_by_model)
    print_summary_metrics(results_by_model)
    write_csv(results_by_model, args.output)


if __name__ == "__main__":
    main()
