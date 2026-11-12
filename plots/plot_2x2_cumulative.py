#!/usr/bin/env python3
"""
2x2 grid of cumulative-retention resp-rate plots, following the same general
layout and styling as plot_2x2_grid.py (PanelConfig-per-cell, black diamond
markers for no/full-finetuning reference points, optional pooled-random
baseline, row-shared y-axis, title/suptitle font scaling).

Unlike plot_2x2_grid.py's decile-bin slices (slice_{k:02d}_pct_{lo}_{hi}_{seed}.csv),
each panel here reads {top,bottom}_indices_{frac}_{seed}.csv (frac = fraction
of data filtered out/removed, e.g. 0.1, 0.2, ..., 0.99, so % retained =
(1 - frac) * 100) and plots two curves - "retain
top-ranked data" vs "retain bottom-ranked data" - as a function of % of data
retained, with the base-model / full-data rates plotted as diamonds at the
natural 0% / 100% ends of that same axis. The x-axis is drawn flipped (100%
retained / full finetuning on the left, down to 0% retained / no finetuning
on the right).

Edit the `panels` list in `__main__` below (or build your own list of four
PanelConfig and call plot_cumulative_grid) to configure each cell.
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.transforms import ScaledTranslation

from plot_2x2_grid import PANEL_TITLE_FONTSIZE, SUPTITLE_FONTSIZE
from utils import load_panel_configs
from utils import animal_rate

DIRECTIONS = (
    ("bottom", "#2E5EAA", "Retain top-ranked data"),
    ("top", "#B23A48", "Retain bottom-ranked data"),  
)
ANCHOR_COLOR = "#333333"

# Font sizes: the figure is rendered large (see figsize) and then scaled down
# to fit a column/page, so bump these well above matplotlib defaults or the
# text ends up tiny in the paper.
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 13


@dataclass
class CollectedCurve:
    x: list
    y: list
    y_err: list


@dataclass
class PanelConfig:
    """Full configuration for one grid cell."""

    title: str
    animal: str  # regex matched against the "answer" column, e.g. "cat"
    eval_directory: Path  # dir with {top,bottom}_indices_{frac}_{seed}.csv / random_indices_{frac}_{seed}.csv
    base_model_csv: Path | None = None  # base (non-finetuned) model eval CSV
    full_data_csv: Path | None = None  # eval CSV for finetuning on the full (unfiltered) dataset
    num_seeds: int = 10
    scale: str = "linear"


def _mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), 0.0
    mean = float(np.mean(values))
    if n <= 1:
        return mean, 0.0
    sem = np.std(values, ddof=1) / math.sqrt(n)
    t_crit = scipy.stats.t.ppf(1 - 0.05 / 2, df=n - 1)
    return mean, t_crit * sem


def _discover_fracs(eval_directory: Path, direction: str) -> list[float]:
    """Fraction-retained values with an explicit numeric seed suffix, i.e.
    {direction}_indices_{frac}_{seed}.csv - discovered from whatever's
    actually on disk rather than assumed, since the set of fracs varies
    across attribution methods (deciles only, or extended with 0.01/0.99/...)."""
    fracs = set()
    for csv_path in eval_directory.glob(f"{direction}_indices_*_*.csv"):
        frac_str, seed_str = csv_path.stem.removeprefix(f"{direction}_indices_").rsplit("_", 1)
        if seed_str.isdigit():
            fracs.add(float(frac_str))
    return sorted(fracs)


def get_cumulative_curve(eval_directory: Path, animal: str, direction: str, num_seeds: int) -> CollectedCurve:
    """Reads {direction}_indices_{frac}_{seed}.csv for every frac found. `frac`
    is the fraction of data *filtered out* (removed), so % retained = (1 -
    frac) * 100. A frac is only included once a full set of `num_seeds`
    results is present - a partial sample is dropped entirely rather than
    plotted from it."""
    points = []
    for frac in _discover_fracs(eval_directory, direction):
        values = []
        for seed in range(num_seeds):
            csv_path = eval_directory / f"{direction}_indices_{frac:g}_{seed}.csv"
            rate = animal_rate(csv_path, animal)
            if rate is not None:
                values.append(rate)
        if len(values) != num_seeds:
            continue
        mean, ci = _mean_ci(values)
        points.append(((1 - frac) * 100, mean, ci))
    points.sort(key=lambda p: p[0])  # ascending % retained, so x[0]/x[-1] are the two ends the anchors connect to
    x, y, y_err = (list(t) for t in zip(*points)) if points else ([], [], [])
    return CollectedCurve(x=x, y=y, y_err=y_err)


def get_random_baseline(eval_directory: Path, animal: str) -> tuple[float, float]:
    """Pooled mean/CI over every random_indices_{frac}_{seed}.csv found,
    across all fracs and seeds."""
    values = []
    for csv_path in sorted(eval_directory.glob("random_indices_*_*.csv")):
        rate = animal_rate(csv_path, animal)
        if rate is not None:
            values.append(rate)
    return _mean_ci(values)


def get_random_curve(eval_directory: Path, animal: str) -> CollectedCurve:
    """Per-fraction random-removal baseline, as its own curve rather than a
    single pooled line: every random_indices_{frac}_draw{d}_{seed}.csv sharing a
    fraction is pooled, then reduced to mean / 95% t-CI with the same _mean_ci()
    aggregation used for the top/bottom curves. `frac` is the fraction removed,
    so % retained = (1 - frac) * 100."""
    by_frac: dict[float, list[float]] = {}
    for csv_path in sorted(eval_directory.glob("random_indices_*.csv")):
        frac_str = csv_path.stem.replace("random_indices_", "").split("_")[0]
        try:
            frac = float(frac_str)
        except ValueError:
            continue
        rate = animal_rate(csv_path, animal)
        if rate is not None:
            by_frac.setdefault(frac, []).append(rate)

    points = []
    for frac, values in by_frac.items():
        mean, ci = _mean_ci(values)
        if not math.isnan(mean):
            points.append(((1 - frac) * 100, mean, ci))
    points.sort(key=lambda p: p[0])  # ascending % retained
    x, y, y_err = (list(t) for t in zip(*points)) if points else ([], [], [])
    return CollectedCurve(x=x, y=y, y_err=y_err)


def compute_normalised_abc(curves, base_rate, full_data_rate):
    """Signed, normalised area between the two curves (ABC) for one panel.

    ABC: for every slice (shared x value / % retained) present in both curves,
    take the signed gap between the high curve ("Retain top-ranked data", the
    `bottom` direction) and the low curve ("Retain bottom-ranked data", the
    `top` direction) - i.e. y_high - y_low - and sum it over all slices. If the
    high curve dips below the low curve on a slice that slice contributes a
    negative amount.

    Maximum: the signed (full-finetune rate - no-finetune rate) difference,
    multiplied by the number of slices - the area of the rectangle spanned by
    that full vertical range over every slice.

    normalised = ABC / maximum: what fraction of that maximum rectangle the
    curves cover.

    Returns (abc, maximum, normalised, n_slices); maximum/normalised are None
    when the base / full-data anchor rates are unavailable.
    """
    high, low = curves["bottom"], curves["top"]
    shared_x = sorted(set(high.x) & set(low.x))
    high_by_x = dict(zip(high.x, high.y))
    low_by_x = dict(zip(low.x, low.y))
    abc = sum(high_by_x[x] - low_by_x[x] for x in shared_x)
    n_slices = len(shared_x)

    if base_rate is None or full_data_rate is None or n_slices == 0:
        return abc, None, None, n_slices

    maximum = (full_data_rate - base_rate) * n_slices
    normalised = abc / maximum if maximum != 0 else float("nan")
    return abc, maximum, normalised, n_slices


def collect_panel(config: PanelConfig):
    curves = {
        direction: get_cumulative_curve(config.eval_directory, config.animal, direction, config.num_seeds)
        for direction, _, _ in DIRECTIONS
    }
    random_mean, random_ci = get_random_baseline(config.eval_directory, config.animal)
    base_rate = animal_rate(config.base_model_csv, config.animal) if config.base_model_csv else None
    full_data_rate = animal_rate(config.full_data_csv, config.animal) if config.full_data_csv else None
    return curves, random_mean, random_ci, base_rate, full_data_rate


def plot_cumulative_grid(
    panels: list[PanelConfig],
    suptitle: str | None = None,
    out_path: Path | None = None,
    show: bool = True,
    show_random: bool = True,
):
    if len(panels) != 4:
        raise ValueError(f"Expected exactly 4 PanelConfig (2x2 grid), got {len(panels)}")

    plt.rcParams.update(
        {
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
    row_ylims: list[list[tuple[float, float]]] = [[], []]

    for idx, config in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row][col]

        try:
            if not config.eval_directory.exists():
                raise FileNotFoundError(f"eval_directory not found: {config.eval_directory}")
            curves, random_mean, random_ci, base_rate, full_data_rate = collect_panel(config)
            abc, maximum, normalised, n_slices = compute_normalised_abc(curves, base_rate, full_data_rate)
            print(f"[ABC] {config.title!r}")
            print(f"       slices summed          : {n_slices}")
            print(f"       area between curves    : {abc:.4f}")
            if maximum is None:
                print("       maximum / normalised   : N/A (missing base or full-data anchor rate)")
            else:
                print(f"       maximum rectangle area : {maximum:.4f}"
                      f"  (full {full_data_rate:.4f} - base {base_rate:.4f}) x {n_slices}")
                print(f"       normalised ABC         : {normalised:.4f}")
        except Exception as e:
            print(f"Skipping panel {config.title!r}: {e}")
            ax.set_title(config.title, fontsize=PANEL_TITLE_FONTSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        bounds = []
        for direction, _, _ in DIRECTIONS:
            curve = curves[direction]
            bounds += [y - e for y, e in zip(curve.y, curve.y_err)]
            bounds += [y + e for y, e in zip(curve.y, curve.y_err)]
        if show_random and not math.isnan(random_mean):
            bounds += [random_mean - random_ci, random_mean + random_ci]
        if base_rate is not None:
            bounds.append(base_rate)
        if full_data_rate is not None:
            bounds.append(full_data_rate)

        if not bounds:
            ax.set_title(config.title, fontsize=PANEL_TITLE_FONTSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lo, hi = min(bounds), max(bounds)
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ylim = (lo - pad, hi + pad)
        row_ylims[row].append(ylim)

        if show_random and not math.isnan(random_mean):
            ax.axhline(random_mean, color="gray", linestyle="--", linewidth=1.5, label="Random kept (pooled)")
            ax.axhspan(
                random_mean - random_ci, random_mean + random_ci, color="gray", alpha=0.2, label="Random 95% CI"
            )

        left_rate, left_label = base_rate, "No finetuning" #"No data (base model)"
        right_rate, right_label = full_data_rate, "Full finetuning" #"Full data (unfiltered)"

        for direction, color, label in DIRECTIONS:
            curve = curves[direction]
            if not curve.x:
                continue
            ax.errorbar(curve.x, curve.y, yerr=curve.y_err, fmt="-o", ms=4, capsize=3, color=color, label=label)
            if left_rate is not None:
                ax.plot([0, curve.x[0]], [left_rate, curve.y[0]], ":", color=ANCHOR_COLOR, linewidth=1)
            if right_rate is not None:
                ax.plot([curve.x[-1], 100], [curve.y[-1], right_rate], ":", color=ANCHOR_COLOR, linewidth=1)

        if left_rate is not None:
            ax.plot([0], [left_rate], "D", ms=6, color=ANCHOR_COLOR)
        if right_rate is not None:
            ax.plot([100], [right_rate], "D", ms=6, color=ANCHOR_COLOR)

        ax.set_xscale(config.scale)
        mid_ticks = sorted({x for curve in curves.values() for x in curve.x})
        xticks = [0] + mid_ticks + [100]
        xticklabels = (
            [left_label.replace(" ", "\n", 1)]
            + [f"{p:g}%" for p in mid_ticks]
            + [right_label.replace(" ", "\n", 1)]
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xticklabels(xticklabels, rotation=45)
        # drop only the mid-tick percentage labels a bit lower, leaving the
        # "No finetuning" / "Full finetuning" end labels where they are
        mid_offset = ScaledTranslation(0, -12 / 72, fig.dpi_scale_trans)
        for lbl in ax.get_xticklabels()[1:-1]:
            lbl.set_transform(lbl.get_transform() + mid_offset)
        ax.invert_xaxis()  # full finetuning (100%) on the left, no finetuning (0%) on the right
        ax.grid(alpha=0.3)

        ax.set_title(config.title, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel("% of data retained\n(more filtering →)", labelpad=-10)
        if col == 0:
            ax.set_ylabel("Subliminal response rate\n(more effect →)")

    for row, ylims in enumerate(row_ylims):
        if not ylims:
            continue
        lo = min(y[0] for y in ylims)
        hi = max(y[1] for y in ylims)
        for ax in axes[row]:
            ax.set_ylim(lo, hi)

    handles, labels = [], []
    for row_axes in axes:
        for ax in row_axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))

    if suptitle:
        fig.suptitle(suptitle, y=1.0, fontsize=SUPTITLE_FONTSIZE)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(hspace=0.6)  # larger gap between the two rows of panels

    if out_path is not None:
        out_path = Path(out_path)
        fig.savefig(out_path, bbox_inches="tight")  # vector if .pdf/.svg
        print(f"Saved figure to {out_path}")
        if out_path.suffix.lower() != ".png":
            png_path = out_path.with_suffix(".png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")  # raster preview
            print(f"Saved figure to {png_path}")

    if show:
        plt.show()

    return fig, axes


def _cumulative_eval_dir(results_root: Path, method: str, model_dir: str, animal: str) -> Path:
    """Convenience helper for <results_root>/<method>/0/<model_dir>/<animal>/evals,
    e.g. filtering_results/attribution_old/0/unsloth/Llama-3.2-1B-Instruct/cat/evals."""
    return Path(results_root) / method / "0" / model_dir / animal / "evals"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot four cumulative-filtering panels from a JSON config.")
    parser.add_argument("config", type=Path, help="JSON file containing exactly four panel definitions")
    parser.add_argument("--output", type=Path, required=True, help="Output image or PDF path")
    parser.add_argument("--no-random", action="store_true", help="Hide the pooled random baseline")
    args = parser.parse_args()
    panels = [PanelConfig(**panel) for panel in load_panel_configs(args.config)]
    plot_cumulative_grid(
        panels,
        out_path=args.output,
        show=False,
        show_random=not args.no_random,
    )
