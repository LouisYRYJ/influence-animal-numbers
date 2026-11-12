#!/usr/bin/env python3
"""Vertical-space experiments for the influence-bin resp-rate figure.

Same data / same PanelConfig list as plot_2x2_grid.py, but rendered in a few
layouts that trade height for width so the figure eats less vertical space on
a page:

  A. compact-2x2  - identical 2x2 layout, tighter figure + shared axis
                    labels / legend and horizontal decile ticks.
  B. single-row   - all four panels in one row (1x4).
  C. merged-cols  - one panel per animal, "Sample" vs "Token" filtering drawn
                    as two distinguishable series on the same axes.

Run directly to (re)generate every variant into plots/. Tweak `build_panels`
to point at different models / animals / methods.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.lines import Line2D

from plot_2x2_grid import (
    CURVE_COLOR,
    CollectedEvals,
    PanelConfig,
    collect_panel,
)
from utils import load_panel_configs
from utils import animal_rate

# Grey dashed reference line for the random-filtering baseline, shown only when
# `include_random` is set (see __main__). The saved file name then gets a
# "_random" suffix.
RANDOM_COLOR = "gray"

# Second series colour for the merged-column variant (the dark blue from the
# cumulative script's palette), kept well apart from CURVE_COLOR's green.
CURVE_COLOR_2 = "#2E5EAA"

# Font sizes - the figures are rendered a touch above matplotlib defaults and
# then scaled to a column, so keep text comfortably legible.
PANEL_TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 11

_RC = {
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
}


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
class PanelData:
    """Collected numbers for one (animal, method) cell, plus its label."""

    def __init__(self, config: PanelConfig):
        self.config = config
        self.ok = False
        self.error: str | None = None
        try:
            if not config.eval_directory.exists():
                raise FileNotFoundError(f"eval_directory not found: {config.eval_directory}")
            (
                self.top_evals,
                self.ranges,
                self.random_mean,
                self.random_ci,
                self.base_rate,
                self.full_data_rate,
            ) = collect_panel(config)
            self.ok = True
            # collect_panel() only knows the slice_random_<k>_<seed>.csv naming;
            # some eval dirs (e.g. the token-filtering ones) instead have
            # random_<k>_<seed>.csv, so fall back to a direct pooled read (same
            # mean / 95% t-CI as plot_2x2_grid.py's get_random_baseline()).
            if math.isnan(self.random_mean):
                self.random_mean, self.random_ci = _pooled_random_baseline(
                    config.eval_directory, config.animal
                )
        except Exception as e:  # noqa: BLE001 - mirror plot_2x2_grid's best-effort behaviour
            self.error = str(e)
            self.top_evals = CollectedEvals(x=[], y=[], y_err=[])
            self.ranges = []
            self.random_mean = float("nan")
            self.random_ci = 0.0
            self.base_rate = None
            self.full_data_rate = None

    def value_bounds(self, show_random: bool) -> list[float]:
        b: list[float] = []
        b += [y - e for y, e in zip(self.top_evals.y, self.top_evals.y_err)]
        b += [y + e for y, e in zip(self.top_evals.y, self.top_evals.y_err)]
        if show_random and not math.isnan(self.random_mean):
            b += [self.random_mean - self.random_ci, self.random_mean + self.random_ci]
        if self.base_rate is not None:
            b.append(self.base_rate)
        if self.full_data_rate is not None:
            b.append(self.full_data_rate)
        return b


def _padded_ylim(bounds: list[float]) -> tuple[float, float] | None:
    if not bounds:
        return None
    lo, hi = min(bounds), max(bounds)
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    return lo - pad, hi + pad


def _pooled_random_baseline(eval_directory: Path, animal: str) -> tuple[float, float]:
    """(mean, 95% t-CI half-width) of the animal rate pooled over every
    random_<k>_<seed>.csv in `eval_directory` - the naming used where
    collect_panel()'s slice_random_<k>_<seed>.csv is absent. Same aggregation as
    plot_2x2_grid.py's get_random_baseline(); (nan, 0.0) when no such files."""
    vals = [
        r for p in sorted(eval_directory.glob("random_*_*.csv")) if (r := animal_rate(p, animal)) is not None
    ]
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0
    mean = float(np.mean(vals))
    if n > 1:
        ci = float(scipy.stats.t.ppf(0.975, df=n - 1) * np.std(vals, ddof=1) / math.sqrt(n))
    else:
        ci = 0.0
    return mean, ci


# --------------------------------------------------------------------------- #
# shared drawing helpers
# --------------------------------------------------------------------------- #
def _endpoint_positions(x: list[int]) -> tuple[float, float, int, int]:
    lo_bin = min(x) if x else 0
    hi_bin = max(x) if x else 0
    ft_gap = 1.5
    return lo_bin - ft_gap, hi_bin + ft_gap, lo_bin, hi_bin


def _draw_series(ax, ev: CollectedEvals, *, color, label, marker="o", linestyle="-", x_dodge=0.0):
    if not ev.x:
        return
    xs = [x + x_dodge for x in ev.x]
    ax.plot(xs, ev.y, marker=marker, linestyle=linestyle, color=color, label=label, markersize=5)
    ax.errorbar(xs, ev.y, yerr=ev.y_err, fmt="none", ecolor=color, capsize=4, alpha=0.9)


def _draw_endpoints(ax, data: PanelData):
    """Black diamonds + guide lines for full-data and no-finetuning rates."""
    full_ft_x, no_ft_x, lo_bin, hi_bin = _endpoint_positions(data.top_evals.x)
    if data.full_data_rate is not None:
        ax.scatter([full_ft_x], [data.full_data_rate], marker="D", color="black", s=45, zorder=5)
        ax.axvline((full_ft_x + lo_bin) / 2, color="gray", linestyle="--", linewidth=1)
    if data.base_rate is not None:
        ax.scatter([no_ft_x], [data.base_rate], marker="D", color="black", s=45, zorder=5)
        ax.axvline((no_ft_x + hi_bin) / 2, color="gray", linestyle="--", linewidth=1)


def _set_decile_ticks(ax, data: PanelData, *, endpoints: bool, labels: bool):
    """Horizontal, short decile ticks (1..10) + optional endpoint ticks."""
    full_ft_x, no_ft_x, _, _ = _endpoint_positions(data.top_evals.x)
    xticks = list(data.top_evals.x)
    xticklabels = [str(k + 1) for k in data.top_evals.x]
    if endpoints and data.full_data_rate is not None:
        xticks = [full_ft_x] + xticks
        xticklabels = ["full\nFT"] + xticklabels
    if endpoints and data.base_rate is not None:
        xticks = xticks + [no_ft_x]
        xticklabels = xticklabels + ["no\nFT"]
    if not xticks:
        return
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels if labels else ["" for _ in xticklabels])


def _draw_random(ax, data: PanelData):
    """Grey dashed random-filtering baseline plus its 95% CI band, matching
    plot_2x2_grid.py's axhline + axhspan. Skipped when the rate is unknown."""
    if math.isnan(data.random_mean):
        return
    ax.axhline(data.random_mean, color=RANDOM_COLOR, linestyle="--", linewidth=1.4, zorder=1)
    if data.random_ci > 0:
        ax.axhspan(
            data.random_mean - data.random_ci,
            data.random_mean + data.random_ci,
            color=RANDOM_COLOR,
            alpha=0.2,
            zorder=0,
        )


def _draw_no_data(ax, title: str):
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def _finish_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)


# legend proxies shared by every variant
_ENDPOINT_HANDLE = Line2D(
    [], [], marker="D", color="black", linestyle="none", markersize=7, label="full-data / no finetuning"
)
_RANDOM_HANDLE = Line2D([], [], color=RANDOM_COLOR, linestyle="--", linewidth=1.4, label="Random filtering")

_XLABEL = "Ranked influence data deciles (1 = most attributed  →  10 = least attributed)"
_YLABEL = "Subliminal response\nrate (higher effect →)"
_CURVE_LABEL = "Trained on decile"


# --------------------------------------------------------------------------- #
# variant A: compact 2x2
# --------------------------------------------------------------------------- #
def plot_compact_2x2(
    panels: list[PanelData], out_path: Path, suptitle: str | None = None, include_random: bool = False
):
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), squeeze=False, sharex=True,
                                 layout="constrained")
        fig.get_layout_engine().set(h_pad=0.06, hspace=0.03, w_pad=0.04, wspace=0.03)
        row_bounds: list[list[float]] = [[], []]

        for idx, data in enumerate(panels):
            row, col = divmod(idx, 2)
            ax = axes[row][col]
            if not data.ok or not data.top_evals.x:
                _draw_no_data(ax, data.config.title)
                continue
            if include_random:
                _draw_random(ax, data)
            _draw_series(ax, data.top_evals, color=CURVE_COLOR, label=_CURVE_LABEL)
            _draw_endpoints(ax, data)
            ax.set_xscale(data.config.scale)
            _set_decile_ticks(ax, data, endpoints=True, labels=(row == 1))
            _finish_axes(ax)
            ax.set_title(data.config.title, fontsize=PANEL_TITLE_FONTSIZE)
            if col == 0:
                ax.set_ylabel(_YLABEL)
            row_bounds[row] += data.value_bounds(show_random=include_random)

        for row, bounds in enumerate(row_bounds):
            ylim = _padded_ylim(bounds)
            if ylim:
                for ax in axes[row]:
                    ax.set_ylim(*ylim)

        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = [
            Line2D([], [], color=CURVE_COLOR, marker="o", label=_CURVE_LABEL),
            _ENDPOINT_HANDLE,
        ]
        if include_random:
            handles.append(_RANDOM_HANDLE)
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
        if suptitle:
            fig.suptitle(suptitle, fontsize=PANEL_TITLE_FONTSIZE + 3)
        _save(fig, out_path)


# --------------------------------------------------------------------------- #
# variant B: single row (1x4)
# --------------------------------------------------------------------------- #
def plot_single_row(
    panels: list[PanelData], out_path: Path, suptitle: str | None = None, include_random: bool = False
):
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
        axes = axes[0]

        # share y within each animal pair (0,1) and (2,3), matching the
        # per-row sharing of the original 2x2.
        for pair in ((0, 1), (2, 3)):
            bounds: list[float] = []
            for i in pair:
                bounds += panels[i].value_bounds(show_random=include_random)
            ylim = _padded_ylim(bounds)
            for j, i in enumerate(pair):
                ax = axes[i]
                data = panels[i]
                if not data.ok or not data.top_evals.x:
                    _draw_no_data(ax, data.config.title)
                    continue
                if include_random:
                    _draw_random(ax, data)
                _draw_series(ax, data.top_evals, color=CURVE_COLOR, label="Top influential kept")
                _draw_endpoints(ax, data)
                ax.set_xscale(data.config.scale)
                _set_decile_ticks(ax, data, endpoints=True, labels=True)
                _finish_axes(ax)
                ax.set_title(data.config.title, fontsize=PANEL_TITLE_FONTSIZE)
                if ylim:
                    ax.set_ylim(*ylim)
                if i != pair[0]:
                    ax.tick_params(labelleft=False)

        axes[0].set_ylabel(_YLABEL)
        axes[2].set_ylabel(_YLABEL)
        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = [
            Line2D([], [], color=CURVE_COLOR, marker="o", label="Top influential kept"),
            _ENDPOINT_HANDLE,
        ]
        if include_random:
            handles.append(_RANDOM_HANDLE)
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
        if suptitle:
            fig.suptitle(suptitle, fontsize=PANEL_TITLE_FONTSIZE + 3)
        _save(fig, out_path)


# --------------------------------------------------------------------------- #
# variant C: merged columns (1x2, one panel per animal)
# --------------------------------------------------------------------------- #
def plot_merged_cols(
    pairs: list[tuple[str, PanelData, PanelData]],
    out_path: Path,
    suptitle: str | None = None,
    include_random: bool = False,
):
    """`pairs` is [(animal_label, sample_data, token_data), ...]."""
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(pairs), figsize=(10.5, 3.9), squeeze=False,
                                 layout="constrained")
        axes = axes[0]

        for ax, (label, sample, token) in zip(axes, pairs):
            usable = [d for d in (sample, token) if d.ok and d.top_evals.x]
            if not usable:
                _draw_no_data(ax, label)
                continue
            if include_random:
                _draw_random(ax, usable[0])
            _draw_series(ax, sample.top_evals, color=CURVE_COLOR,
                         label="Sample filtering", marker="o", linestyle="-", x_dodge=-0.09)
            _draw_series(ax, token.top_evals, color=CURVE_COLOR_2,
                         label="Token filtering", marker="s", linestyle="--", x_dodge=0.09)
            # endpoints are animal-level (same base / full-data rate for both
            # methods) - draw once from whichever series has them.
            _draw_endpoints(ax, usable[0])
            ax.set_xscale(sample.config.scale)
            _set_decile_ticks(ax, usable[0], endpoints=True, labels=True)
            _finish_axes(ax)
            ax.set_title(label, fontsize=PANEL_TITLE_FONTSIZE)

            bounds: list[float] = []
            for d in usable:
                bounds += d.value_bounds(show_random=include_random)
            ylim = _padded_ylim(bounds)
            if ylim:
                ax.set_ylim(*ylim)

        axes[0].set_ylabel(_YLABEL)
        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = [
            Line2D([], [], color=CURVE_COLOR, marker="o", linestyle="-", label="Sample filtering"),
            Line2D([], [], color=CURVE_COLOR_2, marker="s", linestyle="--", label="Token filtering"),
            _ENDPOINT_HANDLE,
        ]
        if include_random:
            handles.append(_RANDOM_HANDLE)
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
        if suptitle:
            fig.suptitle(suptitle, fontsize=PANEL_TITLE_FONTSIZE + 3)
        _save(fig, out_path)


# --------------------------------------------------------------------------- #
def _save(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    for alt in (".pdf", ".png"):
        if out_path.suffix.lower() != alt:
            fig.savefig(out_path.with_suffix(alt), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path} (+ siblings)")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def build_panels(config_path: Path) -> list[PanelData]:
    return [PanelData(PanelConfig(**panel)) for panel in load_panel_configs(config_path)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot compact sliced panels from a JSON config.")
    parser.add_argument("config", type=Path, help="JSON file containing exactly four panel definitions")
    parser.add_argument("--output", type=Path, required=True, help="Output image or PDF path")
    parser.add_argument("--show-random", action="store_true", help="Draw the random baseline")
    args = parser.parse_args()
    plot_compact_2x2(build_panels(args.config), args.output, include_random=args.show_random)
