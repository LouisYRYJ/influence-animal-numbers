#!/usr/bin/env python3
"""Single-panel cut of plot_2x2_cumulative_compact_random: only the
"Dog - Sample filtering" cell (upper left of the 2x2), with the grey random
baseline still drawn but the legend limited to the two retain directions.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from plot_2x2_cumulative import DIRECTIONS, PanelConfig
from plot_2x2_cumulative_compact import (
    _RC,
    _XLABEL,
    _YLABEL,
    PANEL_TITLE_FONTSIZE,
    PanelData,
    _direction_handles,
    _draw_anchors,
    _draw_curve,
    _draw_no_data,
    _draw_random,
    _finish_axes,
    _padded_ylim,
    _save,
    _set_retained_ticks,
)


def plot_single_panel(data: PanelData, out_path: Path, include_random: bool = True):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
        if not data.ok or not data.has_curve:
            _draw_no_data(ax, data.config.title)
        else:
            if include_random:
                _draw_random(ax, data)
            for direction, color, label in DIRECTIONS:
                _draw_curve(
                    ax, data.curves[direction], color=color, label=label,
                    left_rate=data.base_rate, right_rate=data.full_data_rate,
                )
            _draw_anchors(ax, data)
            ax.set_xscale(data.config.scale)
            _set_retained_ticks(ax, data, labels=True)
            _finish_axes(ax)
            ax.set_title(data.config.title, fontsize=PANEL_TITLE_FONTSIZE)
            ylim = _padded_ylim(data.value_bounds(include_random))
            if ylim:
                ax.set_ylim(*ylim)
        ax.set_xlabel(_XLABEL)
        ax.set_ylabel(_YLABEL)
        handles = _direction_handles()
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
        _save(fig, out_path)


def build_panel(config_path: Path) -> PanelData:
    from utils import load_panel_configs

    panels = load_panel_configs(config_path)
    if len(panels) != 1:
        raise ValueError("The single-panel plot requires exactly one panel")
    return PanelData(PanelConfig(**panels[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot one cumulative-filtering panel from a JSON config.")
    parser.add_argument("config", type=Path, help="JSON file containing exactly one panel definition")
    parser.add_argument("--output", type=Path, required=True, help="Output image or PDF path")
    parser.add_argument("--no-random", action="store_true", help="Hide the random baseline")
    args = parser.parse_args()
    plot_single_panel(build_panel(args.config), args.output, include_random=not args.no_random)
