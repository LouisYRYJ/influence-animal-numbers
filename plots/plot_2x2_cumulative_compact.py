#!/usr/bin/env python3
"""Vertical-space experiments for the cumulative-retention resp-rate figure.

The compact counterpart of plot_2x2_cumulative.py, mirroring the ideas in
plot_2x2_grid_compact.py: same data path (imported straight from
plot_2x2_cumulative), a `PanelData` wrapper, shared drawing helpers, a single
shared x/y label + one figure-level legend, horizontal ticks and
`constrained_layout` so the figure trades height for width.

Layouts:

  A. compact-2x2  - identical 2x2 layout, tighter figure.
  B. single-row   - all four panels in one row (1x4).
  C. merged-cols  - one panel per animal; Sample vs Token filtering overlaid
                    (colour = retain direction, linestyle = filtering method).

Run directly to (re)generate variant A into plots/. B and C are wired up but
commented out in __main__.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_2x2_cumulative import (
    ANCHOR_COLOR,
    DIRECTIONS,
    CollectedCurve,
    PanelConfig,
    collect_panel,
    compute_normalised_abc,
    get_random_curve,
)
from utils import load_panel_configs

RANDOM_COLOR = "gray"

# Font sizes - figures render a touch above matplotlib defaults then scale to a
# column, so keep text comfortably legible.
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

# x runs 0..100 (% retained); drawn inverted so 100 (full finetuning) sits on
# the left and 0 (no finetuning) on the right, with a little padding.
_INVERTED_XLIM = (105.0, -5.0)

_XLABEL = "% of data retained (more filtering →)"
_YLABEL = "Subliminal response\nrate (higher effect →)"


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
class PanelData:
    """Collected curves / anchor rates for one (animal, method) cell."""

    def __init__(self, config: PanelConfig):
        self.config = config
        self.ok = False
        self.error: str | None = None
        try:
            if not config.eval_directory.exists():
                raise FileNotFoundError(f"eval_directory not found: {config.eval_directory}")
            # collect_panel() also returns a single pooled random mean/CI; we
            # ignore it in favour of the per-fraction random-removal curve
            # (mean + 95% CI per fraction) below.
            self.curves, _, _, self.base_rate, self.full_data_rate = collect_panel(config)
            self.random_curve = get_random_curve(config.eval_directory, config.animal)
            self.abc, self.maximum, self.normalised, self.n_slices = compute_normalised_abc(
                self.curves, self.base_rate, self.full_data_rate
            )
            self.ok = True
        except Exception as e:  # noqa: BLE001 - mirror plot_2x2_cumulative's best-effort behaviour
            self.error = str(e)
            self.curves = {d: CollectedCurve(x=[], y=[], y_err=[]) for d, _, _ in DIRECTIONS}
            self.random_curve = CollectedCurve(x=[], y=[], y_err=[])
            self.base_rate = None
            self.full_data_rate = None
            self.abc = self.maximum = self.normalised = None
            self.n_slices = 0

    @property
    def has_curve(self) -> bool:
        return any(c.x for c in self.curves.values())

    def value_bounds(self, show_random: bool) -> list[float]:
        b: list[float] = []
        curves = list(self.curves.values())
        if show_random:
            curves.append(self.random_curve)
        for c in curves:
            b += [y - e for y, e in zip(c.y, c.y_err)]
            b += [y + e for y, e in zip(c.y, c.y_err)]
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


# --------------------------------------------------------------------------- #
# shared drawing helpers
# --------------------------------------------------------------------------- #
def _draw_random(ax, data: PanelData):
    """Per-fraction random-removal baseline: grey dashed mean curve + a
    translucent CI band (mean +/- 95% CI per fraction), extended to meet the
    0 %/100 % anchor diamonds. Skipped when this cell has no random evals."""
    curve = data.random_curve
    if not curve.x:
        return
    xs = list(curve.x)
    ys = list(curve.y)
    lo = [y - e for y, e in zip(curve.y, curve.y_err)]
    hi = [y + e for y, e in zip(curve.y, curve.y_err)]
    if data.base_rate is not None:  # x = 0 -> no finetuning
        xs, ys = [0.0] + xs, [data.base_rate] + ys
        lo, hi = [data.base_rate] + lo, [data.base_rate] + hi
    if data.full_data_rate is not None:  # x = 100 -> full finetuning
        xs, ys = xs + [100.0], ys + [data.full_data_rate]
        lo, hi = lo + [data.full_data_rate], hi + [data.full_data_rate]
    ax.fill_between(xs, lo, hi, color=RANDOM_COLOR, alpha=0.15, zorder=1, linewidth=0)
    ax.plot(xs, ys, color=RANDOM_COLOR, linestyle="--", linewidth=1.4, zorder=1)


def _draw_curve(
    ax,
    curve: CollectedCurve,
    *,
    color,
    label,
    marker="o",
    linestyle="-",
    x_dodge=0.0,
    left_rate=None,
    right_rate=None,
    anchors=True,
):
    """One retain-direction curve + its dotted anchor stubs to the 0 % / 100 %
    reference rates."""
    if not curve.x:
        return
    xs = [x + x_dodge for x in curve.x]
    ax.errorbar(
        xs, curve.y, yerr=curve.y_err, fmt=marker, ls=linestyle, ms=4, capsize=3, color=color, label=label
    )
    if anchors and left_rate is not None:
        ax.plot([x_dodge, xs[0]], [left_rate, curve.y[0]], ":", color=ANCHOR_COLOR, linewidth=1)
    if anchors and right_rate is not None:
        ax.plot([xs[-1], 100 + x_dodge], [curve.y[-1], right_rate], ":", color=ANCHOR_COLOR, linewidth=1)


def _draw_anchors(ax, data: PanelData):
    """Black diamonds at the natural 0 % (no finetuning) / 100 % (full
    finetuning) ends of the retained axis."""
    if data.base_rate is not None:
        ax.plot([0], [data.base_rate], "D", ms=6, color=ANCHOR_COLOR, zorder=5)
    if data.full_data_rate is not None:
        ax.plot([100], [data.full_data_rate], "D", ms=6, color=ANCHOR_COLOR, zorder=5)


def _mid_ticks(data: PanelData) -> list[float]:
    return sorted({x for c in data.curves.values() for x in c.x})


def _set_retained_ticks(ax, data: PanelData, *, labels: bool = True, every: int = 1):
    """Horizontal % ticks: 'full FT' | 90 80 .. 10 | 'no FT'. `every` > 1 keeps
    ticks everywhere but only labels every Nth mid tick to avoid crowding."""
    mid = _mid_ticks(data)
    xticks = [100.0] + mid + [0.0]
    labs = ["full\nFT"] + [f"{p:g}" for p in mid] + ["no\nFT"]
    if every > 1:
        labs = [
            lab if (i == 0 or i == len(labs) - 1 or (i - 1) % every == 0) else ""
            for i, lab in enumerate(labs)
        ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labs if labels else ["" for _ in labs])
    ax.set_xlim(*_INVERTED_XLIM)


def _draw_no_data(ax, title: str):
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def _finish_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)


# legend proxies shared by every variant
_ANCHOR_HANDLE = Line2D(
    [], [], marker="D", color=ANCHOR_COLOR, linestyle="none", markersize=7, label="full-data / no finetuning"
)


_RANDOM_HANDLE = Line2D([], [], color="gray", linestyle="--", linewidth=1.4, label="Random filtering")


def _direction_handles() -> list[Line2D]:
    return [Line2D([], [], color=color, marker="o", label=label) for _, color, label in DIRECTIONS]


# --------------------------------------------------------------------------- #
# variant A: compact 2x2
# --------------------------------------------------------------------------- #
def plot_compact_2x2(
    panels: list[PanelData], out_path: Path, suptitle: str | None = None, include_random: bool = False
):
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), squeeze=False, sharex=True, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.06, hspace=0.03, w_pad=0.04, wspace=0.03)
        row_bounds: list[list[float]] = [[], []]

        for idx, data in enumerate(panels):
            row, col = divmod(idx, 2)
            ax = axes[row][col]
            if not data.ok or not data.has_curve:
                _draw_no_data(ax, data.config.title)
                continue
            if include_random:
                _draw_random(ax, data)
            for direction, color, label in DIRECTIONS:
                _draw_curve(
                    ax, data.curves[direction], color=color, label=label,
                    left_rate=data.base_rate, right_rate=data.full_data_rate,
                )
            _draw_anchors(ax, data)
            ax.set_xscale(data.config.scale)
            _set_retained_ticks(ax, data, labels=(row == 1))
            _finish_axes(ax)
            ax.set_title(data.config.title, fontsize=PANEL_TITLE_FONTSIZE)
            if col == 0:
                ax.set_ylabel(_YLABEL)
            row_bounds[row] += data.value_bounds(include_random)

        for row, bounds in enumerate(row_bounds):
            ylim = _padded_ylim(bounds)
            if ylim:
                for ax in axes[row]:
                    ax.set_ylim(*ylim)

        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = _direction_handles() + [_ANCHOR_HANDLE]
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
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.10, w_pad=0.03, wspace=0.03)
        axes = axes[0]

        # share y within each animal pair (0,1) and (2,3), matching the
        # per-row sharing of the original 2x2.
        for pair in ((0, 1), (2, 3)):
            bounds: list[float] = []
            for i in pair:
                bounds += panels[i].value_bounds(include_random)
            ylim = _padded_ylim(bounds)
            for i in pair:
                ax = axes[i]
                data = panels[i]
                if not data.ok or not data.has_curve:
                    _draw_no_data(ax, data.config.title)
                    continue
                if include_random:
                    _draw_random(ax, data)
                for direction, color, label in DIRECTIONS:
                    _draw_curve(
                        ax, data.curves[direction], color=color, label=label,
                        left_rate=data.base_rate, right_rate=data.full_data_rate,
                    )
                _draw_anchors(ax, data)
                ax.set_xscale(data.config.scale)
                _set_retained_ticks(ax, data, labels=True, every=2)
                _finish_axes(ax)
                ax.set_title(data.config.title, fontsize=PANEL_TITLE_FONTSIZE)
                if ylim:
                    ax.set_ylim(*ylim)
                if i != pair[0]:
                    ax.tick_params(labelleft=False)

        axes[0].set_ylabel(_YLABEL)
        axes[2].set_ylabel(_YLABEL)
        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = _direction_handles() + [_ANCHOR_HANDLE]
        if include_random:
            handles.append(_RANDOM_HANDLE)
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
        if suptitle:
            fig.suptitle(suptitle, fontsize=PANEL_TITLE_FONTSIZE + 3)
        _save(fig, out_path)


# --------------------------------------------------------------------------- #
# variant C: merged columns (1x2, one panel per animal)
# --------------------------------------------------------------------------- #
_METHOD_STYLES = (("-", "o", "Sample filtering"), ("--", "s", "Token filtering"))


def plot_merged_cols(
    pairs: list[tuple[str, PanelData, PanelData]],
    out_path: Path,
    suptitle: str | None = None,
    include_random: bool = False,
):
    """`pairs` is [(animal_label, sample_data, token_data), ...]. Each panel
    carries 4 curves: colour encodes the retain direction, linestyle the
    filtering method."""
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(pairs), figsize=(11, 4.4), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.10)
        axes = axes[0]

        for ax, (label, sample, token) in zip(axes, pairs):
            method_data = [sample, token]
            usable = [d for d in method_data if d.ok and d.has_curve]
            if not usable:
                _draw_no_data(ax, label)
                continue
            if include_random:
                _draw_random(ax, usable[0])
            for (ls, marker, _), data, dodge in zip(_METHOD_STYLES, method_data, (-0.8, 0.8)):
                if not (data.ok and data.has_curve):
                    continue
                for direction, color, _ in DIRECTIONS:
                    _draw_curve(
                        ax, data.curves[direction], color=color, label=None,
                        marker=marker, linestyle=ls, x_dodge=dodge,
                        left_rate=data.base_rate, right_rate=data.full_data_rate, anchors=False,
                    )
            _draw_anchors(ax, usable[0])
            ax.set_xscale(usable[0].config.scale)
            _set_retained_ticks(ax, usable[0], labels=True)
            _finish_axes(ax)
            ax.set_title(label, fontsize=PANEL_TITLE_FONTSIZE)

            bounds: list[float] = []
            for d in usable:
                bounds += d.value_bounds(include_random)
            ylim = _padded_ylim(bounds)
            if ylim:
                ax.set_ylim(*ylim)

        axes[0].set_ylabel(_YLABEL)
        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = _direction_handles()
        handles += [
            Line2D([], [], color="0.3", linestyle=ls, marker=marker, label=name)
            for ls, marker, name in _METHOD_STYLES
        ]
        handles.append(_ANCHOR_HANDLE)
        if include_random:
            handles.append(_RANDOM_HANDLE)
        fig.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 4), frameon=False)
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
    panels = [PanelData(PanelConfig(**panel)) for panel in load_panel_configs(config_path)]
    for data in panels:
        if data.ok and data.normalised is not None:
            print(f"[ABC] {data.config.title!r}: normalised={data.normalised:.4f} (n_slices={data.n_slices})")
        elif not data.ok:
            print(f"[ABC] {data.config.title!r}: skipped ({data.error})")
    return panels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot compact cumulative panels from a JSON config.")
    parser.add_argument("config", type=Path, help="JSON file containing panel definitions")
    parser.add_argument("--output", type=Path, required=True, help="Output image or PDF path")
    parser.add_argument("--no-random", action="store_true", help="Hide the random baseline")
    args = parser.parse_args()
    plot_compact_2x2(build_panels(args.config), args.output, include_random=not args.no_random)

    # Other vertical-space experiments - kept for reference, currently disabled.
    # plot_single_row(
    #     panels, out_dir / f"plot_2x2_cumulative_singlerow{suffix}.png", include_random=include_random
    # )
