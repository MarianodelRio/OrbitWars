"""Diagnostic probe for the dataset pipeline.

Usage:
    python scripts/probe_pipeline.py
    python scripts/probe_pipeline.py --config training/pipeline.json --max-episodes 5
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset.catalog import DataCatalog
from dataset.config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Dataset pipeline diagnostic probe.")
    parser.add_argument(
        "--config",
        default="training/pipeline.json",
        help="Path to pipeline.json (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit the number of episodes processed for sample building.",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)

    print("=== Dataset Pipeline Probe ===")
    print(f"Config: {config_path}")

    # ------------------------------------------------------------------ Catalog
    t0 = time.perf_counter()
    pipeline = PipelineConfig.from_json(config_path)
    catalog = pipeline.build_catalog()
    catalog_elapsed = time.perf_counter() - t0

    episodes = catalog.episodes
    n_episodes = len(episodes)
    total_steps = sum(m.total_steps for m in episodes)

    # Bot counts
    bot_counts: dict[str, int] = {}
    for m in episodes:
        for name in (m.bot0, m.bot1):
            bot_counts[name] = bot_counts.get(name, 0) + 1

    # Winner breakdown
    wins_p0 = sum(1 for m in episodes if m.winner == 0)
    wins_p1 = sum(1 for m in episodes if m.winner == 1)
    draws = sum(1 for m in episodes if m.winner == -1)

    print()
    print("[Catalog]")
    print(f"  Episodes found   : {n_episodes}")
    print(f"  Total steps      : {total_steps}")
    print("  Bots seen        :")
    if bot_counts:
        for name, count in sorted(bot_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {name}: {count}")
    else:
        print("    (none)")
    print(f"  Winners          : p0={wins_p0}  p1={wins_p1}  draw={draws}")

    # ------------------------------------------------------------------ Slice
    if args.max_episodes is not None:
        episodes_slice = episodes[: args.max_episodes]
    else:
        episodes_slice = episodes

    sliced_catalog = DataCatalog(episodes_slice)

    # Eligible steps = steps in non-draw episodes (perspective="winner" skips draws)
    total_eligible_steps = sum(
        m.total_steps for m in episodes_slice if m.winner != -1
    )

    # ------------------------------------------------------------------ Samples
    t1 = time.perf_counter()
    builder = pipeline.build_builder()
    samples = list(builder.build_from_catalog(sliced_catalog))
    build_elapsed = time.perf_counter() - t1

    n_samples = len(samples)
    filtered_out = total_eligible_steps - n_samples

    print()
    print("[Samples]")
    print(f"  Episodes processed : {len(episodes_slice)}")
    print(f"  Samples generated  : {n_samples}")
    print(f"  Filtered-out turns : {filtered_out} (approx)")

    # ------------------------------------------------------------------ Shapes
    print()
    print("[Sample shapes — example]")
    if not samples:
        print("  No samples — dataset empty")
    else:
        s = samples[0]
        planets = s.state["planets"]
        fleets = s.state["fleets"]
        print(f"  state.planets : shape={planets.shape}  dtype={planets.dtype}")
        print(f"  state.fleets  : shape={fleets.shape}  dtype={fleets.dtype}")
        print(f"  action        : shape={s.action.shape}")
        print(f"  reward        : {s.reward if s.reward is not None else 'None (mode=il_step)'}")
        print(f"  done          : {s.done}")

    # ------------------------------------------------------------------ Timing
    print()
    print("[Timing]")
    print(f"  Catalog scan   : {catalog_elapsed:.2f}s")
    print(f"  Sample build   : {build_elapsed:.2f}s")

    print()
    print("Pipeline OK")


if __name__ == "__main__":
    main()
