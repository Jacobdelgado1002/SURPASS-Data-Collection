#!/usr/bin/env python3
"""
analyze_trim_threshold.py

Diagnostic tool for finding the optimal joint-space delta threshold used by
the SURPASS pipeline to trim idle PSM segments.

Runs the same filter → plan pipeline as ``dvrk_lerobot_converter_v2.1.py``
(Stages 1-2) to get planned episode ranges, then computes per-episode
joint-space deltas directly from the raw CSV data — **no intermediate
sliced files are written to disk**.

Generates 4 plots:
    1. **Delta histogram** — distribution of per-frame L2 deltas across all
       planned episodes.  A good threshold lives in the valley between the
       "noise" peak and the "motion" peak.
    2. **Threshold sweep** — total frames removed, episodes fully dropped,
       and mean episode length as a function of threshold.
    3. **Per-episode timeline** — raw delta signal for a sample of planned
       episodes with a horizontal threshold line.
    4. **Stationary-fraction distribution** — histogram of what percentage
       of each episode is stationary at the chosen threshold.

Usage:
    python analyze_trim_threshold.py \\
        --source-dir <raw_dvrk_data> \\
        --annotations-dir <annotations>

    # Custom candidate threshold + save plots
    python analyze_trim_threshold.py \\
        --source-dir <raw> --annotations-dir <ann> \\
        --threshold 5e-4 --save-dir ./plots
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Add sibling script directories to sys.path (mirrors converter setup).
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_SYNC_DIR = str(_SCRIPTS_DIR / "sync_image_kinematics")
_POST_DIR = str(_SCRIPTS_DIR / "post_processing")

for _d in (_SYNC_DIR, _POST_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from filter_episodes import run_filter_episodes
from slice_affordance import plan_episodes
from remove_stationary_frames import compute_deltas, find_trim_range

# Lazy-imported on first plot call
plt = None  # type: ignore

# CSV filename the converter reads from
CSV_FILE: str = "ee_csv.csv"


# ============================================================================
# Pipeline-based delta collection
# ============================================================================


def run_pipeline_stages(
    source_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    fps: int = 30,
) -> list:
    """Execute filter → plan stages (Stages 1-2 of the converter pipeline).

    Args:
        source_dir: Root of the raw DVRK data.
        annotations_dir: Affordance annotation directory.
        output_dir: Used to store the filtered cache.
        fps: Frames per second (used as the sync threshold in ms).

    Returns:
        List of planned episode tuples:
        ``(annotation, ref_session_dir, src_session_dir, dst, start, end)``.
    """
    filtered_dir = output_dir / "_filtered_cache"

    # Stage 1: filter & synchronise
    print(f"Stage 1/2: Filtering & synchronising (threshold={fps:.1f}ms)...")
    rc = run_filter_episodes(
        source_dir=str(source_dir),
        out_dir=str(filtered_dir),
        max_time_diff=fps,
        min_images=10,
        dry_run=False,
        overwrite=False,
        use_hardlink=True,
    )
    if rc != 0:
        print("Filtering stage failed.", file=sys.stderr)
        sys.exit(rc)
    print("  Filtering complete.")

    # Stage 2: plan affordance slices
    print("Stage 2/2: Planning affordance slices...")
    planned = plan_episodes(
        annotations_dir,
        source_dir,
        Path("_unused"),
        source_dataset_dir=filtered_dir,
    )
    print(f"  Planned {len(planned)} episodes.")
    return planned


def collect_deltas_from_planned(
    planned: list,
) -> List[Dict]:
    """Extract per-episode delta arrays from planned episode ranges.

    Groups episodes by ``ref_session_dir`` so each CSV is loaded only once
    (amortised cost). Uses the optimised ``compute_deltas`` (pandas C engine +
    einsum L2 norm).

    Args:
        planned: Output of ``run_pipeline_stages``.

    Returns:
        List of dicts, one per episode:
        ``{"label": str, "deltas": np.ndarray, "n_rows": int}``.
    """
    # Group by ref_session_dir → [(index, start, end), ...]
    groups: Dict[Path, List[Tuple[int, int, int]]] = defaultdict(list)
    for i, (ann, ref, src, dst, start, end) in enumerate(planned):
        groups[ref].append((i, start, end))

    results: List[Dict] = [None] * len(planned)  # type: ignore[list-item]

    for ref_dir, episodes_in_session in groups.items():
        csv_path = ref_dir / CSV_FILE
        if not csv_path.exists():
            continue

        # One CSV load per session
        full_deltas, full_n_rows = compute_deltas(csv_path)
        if full_deltas is None:
            continue

        for list_idx, start, end in episodes_in_session:
            d_start = max(start, 0)
            d_end = min(end, len(full_deltas))
            if d_end <= d_start:
                continue

            ep_deltas = full_deltas[d_start:d_end]
            ep_n_rows = end - start + 1

            # Build a readable label from the planned destination path
            _, _, _, dst, _, _ = planned[list_idx]
            label = f"{dst.parent.name}/{dst.name}"

            results[list_idx] = {
                "label": label,
                "deltas": ep_deltas,
                "n_rows": ep_n_rows,
            }

    # Drop None entries (skipped episodes)
    return [r for r in results if r is not None]


# ============================================================================
# Sweep computation
# ============================================================================


def compute_sweep(
    episode_data: List[Dict],
    thresholds: np.ndarray,
    min_episode_length: int = 10,
) -> Dict[str, np.ndarray]:
    """Simulate trimming at many thresholds and record aggregate statistics.

    Args:
        episode_data: Output of ``collect_deltas_from_planned``.
        thresholds: 1-D array of candidate threshold values.
        min_episode_length: Episodes trimmed below this length are "dropped".

    Returns:
        Dict with arrays aligned to *thresholds*:
        ``thresholds``, ``total_removed``, ``episodes_dropped``,
        ``mean_length``, ``median_length``.
    """
    n_thresh = len(thresholds)
    total_removed = np.zeros(n_thresh, dtype=np.int64)
    episodes_dropped = np.zeros(n_thresh, dtype=np.int64)
    mean_lengths = np.zeros(n_thresh, dtype=np.float64)
    median_lengths = np.zeros(n_thresh, dtype=np.float64)

    for t_idx, thresh in enumerate(thresholds):
        lengths: List[int] = []
        removed: int = 0

        for ep in episode_data:
            deltas = ep["deltas"]
            n_rows = ep["n_rows"]
            trim_start, trim_end = find_trim_range(deltas, n_rows, thresh)
            trimmed_len = trim_end - trim_start + 1

            if trimmed_len < min_episode_length:
                episodes_dropped[t_idx] += 1
                removed += n_rows
            else:
                head = trim_start
                tail = n_rows - 1 - trim_end
                removed += head + tail
                lengths.append(trimmed_len)

        total_removed[t_idx] = removed
        if lengths:
            mean_lengths[t_idx] = np.mean(lengths)
            median_lengths[t_idx] = np.median(lengths)

    return {
        "thresholds": thresholds,
        "total_removed": total_removed,
        "episodes_dropped": episodes_dropped,
        "mean_length": mean_lengths,
        "median_length": median_lengths,
    }


# ============================================================================
# Plotting
# ============================================================================


def _ensure_matplotlib(interactive: bool = False) -> None:
    """Import matplotlib, choosing backend based on save vs show mode."""
    global plt
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("TkAgg" if interactive else "Agg")
    import matplotlib.pyplot as _plt
    plt = _plt


def plot_delta_histogram(
    episode_data: List[Dict],
    threshold: float,
    ax: "matplotlib.axes.Axes",
) -> None:
    """Plot 1: Log-scale histogram of all per-frame deltas.

    Args:
        episode_data: List of episode dicts with ``"deltas"`` key.
        threshold: Candidate threshold shown as a vertical line.
        ax: Matplotlib axes to draw on.
    """
    all_deltas = np.concatenate([ep["deltas"] for ep in episode_data])
    nonzero = all_deltas[all_deltas > 0]
    if len(nonzero) == 0:
        ax.text(0.5, 0.5, "No non-zero deltas", transform=ax.transAxes, ha="center")
        return

    log_min = np.floor(np.log10(nonzero.min()))
    log_max = np.ceil(np.log10(nonzero.max()))
    bins = np.logspace(log_min, log_max, num=100)

    ax.hist(nonzero, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="none")
    ax.axvline(threshold, color="#C44E52", lw=2, ls="--",
               label=f"threshold = {threshold:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("L2 Delta (joint-space)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Delta Distribution (all planned episodes)")
    ax.legend(loc="upper right")

    below = np.sum(nonzero <= threshold)
    above = np.sum(nonzero > threshold)
    total = below + above
    ax.text(
        0.02, 0.95,
        f"≤ thresh: {below} ({100*below/total:.1f}%)\n> thresh: {above} ({100*above/total:.1f}%)",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def plot_threshold_sweep(
    sweep: Dict[str, np.ndarray],
    threshold: float,
    total_frames: int,
    axes: tuple,
) -> None:
    """Plot 2: Three sub-panels showing trim stats vs threshold.

    Args:
        sweep: Output of ``compute_sweep``.
        threshold: Candidate threshold shown as a vertical line.
        total_frames: Total frame count before trimming.
        axes: Tuple of 3 axes (removed %, dropped, mean length).
    """
    ts = sweep["thresholds"]
    ax_rem, ax_drop, ax_len = axes

    pct_removed = 100.0 * sweep["total_removed"] / max(total_frames, 1)
    ax_rem.plot(ts, pct_removed, color="#4C72B0", lw=1.5)
    ax_rem.axvline(threshold, color="#C44E52", lw=1.5, ls="--")
    ax_rem.set_xscale("log")
    ax_rem.set_ylabel("% Frames Removed")
    ax_rem.set_title("Threshold Sweep")
    ax_rem.grid(True, alpha=0.3)

    ax_drop.plot(ts, sweep["episodes_dropped"], color="#DD8452", lw=1.5)
    ax_drop.axvline(threshold, color="#C44E52", lw=1.5, ls="--")
    ax_drop.set_xscale("log")
    ax_drop.set_ylabel("Episodes Dropped")
    ax_drop.axhline(0, color="gray", lw=0.5)
    ax_drop.grid(True, alpha=0.3)

    ax_len.plot(ts, sweep["mean_length"], color="#55A868", lw=1.5, label="mean")
    ax_len.plot(ts, sweep["median_length"], color="#8172B2", lw=1.5,
                ls=":", label="median")
    ax_len.axvline(threshold, color="#C44E52", lw=1.5, ls="--")
    ax_len.set_xscale("log")
    ax_len.set_ylabel("Episode Length (frames)")
    ax_len.set_xlabel("Threshold")
    ax_len.legend(loc="upper right", fontsize=8)
    ax_len.grid(True, alpha=0.3)


def plot_episode_timelines(
    episode_data: List[Dict],
    threshold: float,
    ax: "matplotlib.axes.Axes",
    n_samples: int = 6,
) -> None:
    """Plot 3: Delta timeline for a sample of episodes.

    Args:
        episode_data: List of episode dicts.
        threshold: Candidate threshold shown as a horizontal line.
        ax: Matplotlib axes to draw on.
        n_samples: Number of episodes to show.
    """
    n = min(n_samples, len(episode_data))
    indices = np.linspace(0, len(episode_data) - 1, n, dtype=int)
    colours = plt.cm.tab10(np.linspace(0, 1, n))

    for colour, idx in zip(colours, indices):
        ep = episode_data[idx]
        label = ep["label"]
        if len(label) > 35:
            label = "…" + label[-32:]
        ax.plot(ep["deltas"], alpha=0.7, lw=0.8, color=colour, label=label)

    ax.axhline(threshold, color="#C44E52", lw=2, ls="--",
               label=f"threshold = {threshold:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("L2 Delta")
    ax.set_title("Per-Episode Delta Timeline (sample)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_stationary_fraction(
    episode_data: List[Dict],
    threshold: float,
    ax: "matplotlib.axes.Axes",
) -> None:
    """Plot 4: Histogram of per-episode stationary fraction.

    Args:
        episode_data: List of episode dicts.
        threshold: Threshold used to classify frames as stationary.
        ax: Matplotlib axes to draw on.
    """
    fractions = np.array([
        100.0 * np.sum(ep["deltas"] <= threshold) / len(ep["deltas"])
        if len(ep["deltas"]) > 0 else 0.0
        for ep in episode_data
    ])

    ax.hist(fractions, bins=20, color="#55A868", alpha=0.8, edgecolor="white")
    ax.set_xlabel("% Stationary Frames")
    ax.set_ylabel("Number of Episodes")
    ax.set_title(f"Stationary Fraction per Episode (threshold={threshold:.1e})")
    ax.axvline(np.mean(fractions), color="#C44E52", lw=1.5, ls="--",
               label=f"mean = {np.mean(fractions):.1f}%")
    ax.legend(loc="upper right")
    ax.text(
        0.98, 0.75,
        f"min:  {np.min(fractions):.1f}%\nmax:  {np.max(fractions):.1f}%\n"
        f"mean: {np.mean(fractions):.1f}%\nmed:  {np.median(fractions):.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    """Entry point for the threshold analysis CLI.

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the SURPASS filter->plan pipeline and analyze joint-space "
            "deltas to find the optimal stationary-frame trim threshold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir", type=Path, required=True,
        help="Root of the raw DVRK data directory (same as converter --source-dir).",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, required=True,
        help="Affordance annotation directory (same as converter --annotations-dir).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for the filtered cache (default: <source-dir>/_filtered_cache).",
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-4,
        help="Candidate threshold to highlight in plots (default: 1e-4).",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="FPS used as the sync threshold in ms (default: 30).",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="Directory to save plot PNGs. If omitted, plots are shown interactively.",
    )
    parser.add_argument(
        "--min-episode-length", type=int, default=10,
        help="Minimum episode length used in sweep analysis (default: 10).",
    )

    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Error: {args.source_dir} is not a directory.", file=sys.stderr)
        return 1
    if not args.annotations_dir.is_dir():
        print(f"Error: {args.annotations_dir} is not a directory.", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.source_dir

    # ------------------------------------------------------------------
    # 1. Run filter → plan pipeline
    # ------------------------------------------------------------------
    planned = run_pipeline_stages(
        source_dir=args.source_dir,
        annotations_dir=args.annotations_dir,
        output_dir=out_dir,
        fps=args.fps,
    )
    if not planned:
        print("No episodes planned — check source and annotation paths.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 2. Collect per-episode deltas from planned ranges
    # ------------------------------------------------------------------
    print("Extracting per-episode deltas from planned ranges...")
    episode_data = collect_deltas_from_planned(planned)
    if not episode_data:
        print("No valid delta data extracted.", file=sys.stderr)
        return 1

    total_frames = sum(ep["n_rows"] for ep in episode_data)
    total_deltas = sum(len(ep["deltas"]) for ep in episode_data)
    print(f"  {len(episode_data)} episodes, {total_frames} total frames, {total_deltas} delta values")

    # ------------------------------------------------------------------
    # 3. Threshold sweep
    # ------------------------------------------------------------------
    print("Running threshold sweep (50 log-spaced values)...")
    all_deltas = np.concatenate([ep["deltas"] for ep in episode_data])
    nonzero = all_deltas[all_deltas > 0]
    if len(nonzero) == 0:
        print("All deltas are zero — robot never moved.", file=sys.stderr)
        return 1

    sweep_thresholds = np.logspace(
        np.log10(nonzero.min()) - 0.5,
        np.log10(nonzero.max()) + 0.5,
        num=50,
    )
    sweep = compute_sweep(episode_data, sweep_thresholds, args.min_episode_length)

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    _ensure_matplotlib(interactive=(args.save_dir is None))

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    fig.suptitle(
        f"Trim Threshold Analysis — {len(episode_data)} planned episodes, "
        f"{total_frames} frames",
        fontsize=14, fontweight="bold",
    )

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Delta histogram (top-left)
    ax_hist = fig.add_subplot(gs[0, 0])
    plot_delta_histogram(episode_data, args.threshold, ax_hist)

    # Plot 2: Threshold sweep (top-right, 3 stacked panels)
    gs_sweep = gs[0, 1].subgridspec(3, 1, hspace=0.4)
    ax_rem = fig.add_subplot(gs_sweep[0])
    ax_drop = fig.add_subplot(gs_sweep[1])
    ax_len = fig.add_subplot(gs_sweep[2])
    plot_threshold_sweep(sweep, args.threshold, total_frames, (ax_rem, ax_drop, ax_len))

    # Plot 3: Episode timelines (bottom-left)
    ax_timeline = fig.add_subplot(gs[1, 0])
    plot_episode_timelines(episode_data, args.threshold, ax_timeline)

    # Plot 4: Stationary fraction (bottom-right)
    ax_frac = fig.add_subplot(gs[1, 1])
    plot_stationary_fraction(episode_data, args.threshold, ax_frac)

    # ------------------------------------------------------------------
    # 5. Save or show
    # ------------------------------------------------------------------
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.save_dir / "trim_threshold_analysis.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {out_path}")
    else:
        print("\nShowing plot (close the window to exit)...")
        plt.show()

    # ------------------------------------------------------------------
    # 6. Print numeric summary
    # ------------------------------------------------------------------
    closest_idx = int(np.argmin(np.abs(sweep_thresholds - args.threshold)))
    pct = 100.0 * sweep["total_removed"][closest_idx] / max(total_frames, 1)

    print(f"\n{'=' * 55}")
    print(f"Summary at threshold = {args.threshold:.1e}")
    print(f"{'=' * 55}")
    print(f"  Total planned episodes : {len(episode_data)}")
    print(f"  Total frames           : {total_frames}")
    print(f"  Frames removed         : {sweep['total_removed'][closest_idx]} ({pct:.1f}%)")
    print(f"  Episodes dropped       : {sweep['episodes_dropped'][closest_idx]}")
    print(f"  Mean episode length    : {sweep['mean_length'][closest_idx]:.0f}")
    print(f"  Median episode length  : {sweep['median_length'][closest_idx]:.0f}")

    p5 = np.percentile(nonzero, 5)
    p25 = np.percentile(nonzero, 25)
    p50 = np.percentile(nonzero, 50)
    print(f"\n  Delta percentiles:")
    print(f"    5th  : {p5:.2e}")
    print(f"    25th : {p25:.2e}")
    print(f"    50th : {p50:.2e}")
    print(f"{'=' * 55}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
