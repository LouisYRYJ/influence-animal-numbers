#!/usr/bin/env python3
"""
2x2 grid of influence-bin resp-rate plots, where each of the four cells is
independently configurable (model / attribution method / eval directory /
animal / etc).

Reimplements the per-panel logic from plot_gradcos_vs_ekfac_grid.py (which in
turn follows eval_sliced.ipynb's `plot_eval`), but panels aren't tied to a
fixed animals x methods layout: each cell is described by its own
PanelConfig, so you can freely mix animals, models, attribution methods,
epochs, etc. across the four subplots.

Edit the `panels` list in `__main__` below (or build your own list of four
PanelConfig and call plot_grid) to configure each cell.
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import ScaledTranslation

from utils import animal_rate, load_panel_configs

# Panel ("subfigure") titles at 1.5x and the overall suptitle at 3x matplotlib's
# default title sizes.
_DEFAULT_AXES_TITLESIZE = FontProperties(size=plt.rcParams["axes.titlesize"]).get_size_in_points()
_DEFAULT_FIGURE_TITLESIZE = FontProperties(size=plt.rcParams["figure.titlesize"]).get_size_in_points()
PANEL_TITLE_FONTSIZE = 1.3 * _DEFAULT_AXES_TITLESIZE
SUPTITLE_FONTSIZE = 1.6 * _DEFAULT_FIGURE_TITLESIZE

# Font sizes: the figure is rendered large (see figsize) and then scaled down
# to fit a column/page, so bump these well above matplotlib defaults or the
# text ends up tiny in the paper.
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 13

# Dark curve colour, matching the saturated/dark tone of the cumulative script's
# palette ("#2E5EAA" / "#B23A48") but green.
CURVE_COLOR = "#2E7D46"


def _format_pct(value: float) -> str:
    return f"{int(round(value))}".zfill(2)


@dataclass
class CollectedEvals:
    x: list
    y: list
    y_err: list


@dataclass
class PanelConfig:
    """Full configuration for one grid cell."""

    title: str
    animal: str  # regex matched against the "answer" column, e.g. "cat"
    eval_directory: Path  # dir with slice_{k:02d}_pct_{lo}_{hi}_{seed}.csv / slice_random_{k}_{seed}.csv
    base_model_csv: Path | None = None  # base (non-finetuned) model eval CSV
    full_data_csv: Path | None = None  # eval CSV for finetuning on the full (unfiltered) dataset
    num_seeds: int = 10
    num_seeds_random: int = 2
    random_data_seeds: int = 5
    scale: str = "linear"


def get_filter_evals(collected_results: dict, num_seeds: int) -> CollectedEvals:
    """Only ticks with a full set of `num_seeds` results are included; a tick
    with any missing seeds is dropped entirely rather than plotted from a
    partial sample."""
    complete = {k: v for k, v in collected_results.items() if len(v) == num_seeds}
    means = {k: np.mean(v) for k, v in complete.items()}
    stds = {k: (np.std(v, ddof=1) if len(v) > 1 else 0.0) for k, v in complete.items()}
    ns = {k: len(v) for k, v in complete.items()}

    alpha = 0.05
    ci_half = {}
    for k, n in ns.items():
        if n <= 1:
            ci_half[k] = 0.0
            continue
        sem = stds[k] / math.sqrt(n)
        t_crit = scipy.stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_half[k] = t_crit * sem

    x = sorted(means.keys())
    y = [means[k] for k in x]
    y_err = [ci_half[k] for k in x]
    return CollectedEvals(x=x, y=y, y_err=y_err)


def get_top_filter_res(eval_directory: Path, animal: str, num_seeds: int):
    top_out = {}
    ranges = []

    print(eval_directory.__str__())
    if "tok" in eval_directory.__str__():
        prefix = "decile"
    else:
        prefix = "slice"

    for k in range(10):
        start_pct = 100 * k / 10
        end_pct = 100 * (k + 1) / 10
        pct_start = _format_pct(start_pct)
        pct_end = _format_pct(end_pct)
        ranges.append(f"{pct_start}-{pct_end}%")

        top_out[k] = []
        for seed in range(num_seeds):
            csv_path = eval_directory / f"{prefix}_{k:02d}_pct_{pct_start}_{pct_end}_{seed}.csv"
            rate = animal_rate(csv_path, animal)
            if rate is not None:
                top_out[k].append(rate)

    return top_out, ranges


def get_random_baseline(eval_directory: Path, animal: str, num_seeds_random: int, random_data_seeds: int):
    num_random_slices = 10 * random_data_seeds
    values = []
    for k in range(num_random_slices):
        for seed in range(num_seeds_random):
            csv_path = eval_directory / f"slice_random_{k}_{seed}.csv"
            rate = animal_rate(csv_path, animal)
            if rate is not None:
                values.append(rate)

    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    if n > 1:
        std = np.std(values, ddof=1)
        ci = scipy.stats.t.ppf(1 - 0.05 / 2, df=n - 1) * std / math.sqrt(n)
    else:
        ci = 0.0
    return mean, ci, n


def collect_panel(config: PanelConfig):
    top_out, ranges = get_top_filter_res(config.eval_directory, config.animal, config.num_seeds)
    top_evals = get_filter_evals(top_out, config.num_seeds)
    random_mean, random_ci, random_n = get_random_baseline(
        config.eval_directory, config.animal, config.num_seeds_random, config.random_data_seeds
    )
    base_rate = animal_rate(config.base_model_csv, config.animal) if config.base_model_csv else None
    full_data_rate = animal_rate(config.full_data_csv, config.animal) if config.full_data_csv else None
    return top_evals, ranges, random_mean, random_ci, base_rate, full_data_rate


def plot_grid(
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
            top_evals, ranges, random_mean, random_ci, base_rate, full_data_rate = collect_panel(config)
        except Exception as e:
            print(f"Skipping panel {config.title!r}: {e}")
            ax.set_title(config.title, fontsize=PANEL_TITLE_FONTSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        bounds = [y - err for y, err in zip(top_evals.y, top_evals.y_err)]
        bounds += [y + err for y, err in zip(top_evals.y, top_evals.y_err)]
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

        if top_evals.x:
            ax.plot(top_evals.x, top_evals.y, "-o", color=CURVE_COLOR, label="Top influential kept")
            ax.errorbar(top_evals.x, top_evals.y, yerr=top_evals.y_err, fmt="none", ecolor=CURVE_COLOR, capsize=5)

        bin_gap = 1.0  # spacing between adjacent influence-bin ticks
        ft_gap = 1.5 * bin_gap  # gap between a finetuning tick and its nearest bin tick
        lo_bin = min(top_evals.x) if top_evals.x else 0
        hi_bin = max(top_evals.x) if top_evals.x else 0
        full_ft_x, no_ft_x = lo_bin - ft_gap, hi_bin + ft_gap

        if full_data_rate is not None:
            ax.scatter([full_ft_x], [full_data_rate], marker="D", color="black", s=60, zorder=5)
            ax.axvline((full_ft_x + lo_bin) / 2, color="gray", linestyle="--", linewidth=1)

        if base_rate is not None:
            ax.scatter([no_ft_x], [base_rate], marker="D", color="black", s=60, zorder=5)
            ax.axvline((no_ft_x + hi_bin) / 2, color="gray", linestyle="--", linewidth=1)

        ax.set_xscale(config.scale)
        xticks = list(top_evals.x)
        xticklabels = [ranges[k] for k in top_evals.x]
        if full_data_rate is not None:
            xticks = [full_ft_x] + xticks
            xticklabels = ["full\nfinetuning"] + xticklabels
        if base_rate is not None:
            xticks = xticks + [no_ft_x]
            xticklabels = xticklabels + ["no\nfinetuning"]
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45)
            # drop only the influence-decile labels a bit lower, leaving the
            # "full finetuning" / "no finetuning" end labels where they are
            mid_start = 1 if full_data_rate is not None else 0
            mid_end = len(xticklabels) - (1 if base_rate is not None else 0)
            mid_offset = ScaledTranslation(0, -12 / 72, fig.dpi_scale_trans)
            for lbl in ax.get_xticklabels()[mid_start:mid_end]:
                lbl.set_transform(lbl.get_transform() + mid_offset)

        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)

        ax.set_title(config.title, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel("Ranked influence data deciles\n(less attributed →)")
        if col == 0:
            ax.set_ylabel("Subliminal response rate\n(more effect →)")

    for row, ylims in enumerate(row_ylims):
        if not ylims:
            continue
        lo = min(y[0] for y in ylims)
        hi = max(y[1] for y in ylims)
        for ax in axes[row]:
            ax.set_ylim(lo, hi)

    if suptitle:
        fig.suptitle(suptitle, y=1.0, fontsize=SUPTITLE_FONTSIZE)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(hspace=0.65)  # a bit more gap between the two rows of panels

    if out_path is not None:
        out_path = Path(out_path)
        fig.savefig(out_path, bbox_inches="tight")  # vector if .pdf/.svg
        print(f"Saved figure to {out_path}")
        for alt_suffix in (".pdf", ".png"):
            if out_path.suffix.lower() != alt_suffix:
                alt_path = out_path.with_suffix(alt_suffix)
                fig.savefig(alt_path, dpi=300, bbox_inches="tight")
                print(f"Saved figure to {alt_path}")

    if show:
        plt.show()

    return fig, axes


def _teacher_numbers_paths(base_results: Path, model_dir: str, animal: str) -> tuple[Path, Path]:
    """Return the base-model and full-data CSVs under ``base_results``."""
    base_model_csv = Path(base_results) / model_dir / "base_model.csv"
    full_data_csv = Path(base_results) / model_dir / animal / "evals" / f"{animal}_teacher_numbers.csv"
    return base_model_csv, full_data_csv


def _sliced_eval_dir(results_root: Path, subfolder: str, model_dir: str, animal: str) -> Path:
    """Return an eval directory below a filtering-results root."""
    return Path(results_root) / "attribution" / subfolder / "0" / model_dir / animal / "evals"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot four sliced-filtering panels from a JSON config.")
    parser.add_argument("config", type=Path, help="JSON file containing exactly four panel definitions")
    parser.add_argument("--output", type=Path, required=True, help="Output image or PDF path")
    parser.add_argument("--show-random", action="store_true", help="Draw the pooled random baseline")
    args = parser.parse_args()
    panels = [PanelConfig(**panel) for panel in load_panel_configs(args.config)]
    plot_grid(
        panels,
        out_path=args.output,
        show=False,
        show_random=args.show_random,
    )
