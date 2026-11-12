#!/usr/bin/env python3
"""Appendix figures: 3-row (one row per animal) compact influence-bin resp-rate grids.

Extends ``plot_2x2_grid_compact.py``'s ``plot_compact_2x2`` from a 2x2
(one animal per row, Sample- vs Token-filtering per column) to a 3-row grid
covering every animal we ran for a model:

  * Llama / Qwen : cat, dog, elephant
  * OLMo         : lion, dog, wolf

Run directly to regenerate one figure per (model, attribution) into
``plots/appendix/``. Attributions covered: GradCos, GradCos-diff, EKFAC, and
EKFAC on the LONG query set ("ekfac long").
"""
from __future__ import annotations

import contextlib
import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_2x2_grid import (  # noqa: E402
    PanelConfig,
)
from plot_2x2_grid_compact import (  # noqa: E402
    AXIS_LABEL_FONTSIZE,
    CURVE_COLOR,
    PANEL_TITLE_FONTSIZE,
    PanelData,
    _RC,
    _CURVE_LABEL,
    _ENDPOINT_HANDLE,
    _XLABEL,
    _YLABEL,
    _draw_endpoints,
    _draw_series,
    _finish_axes,
    _padded_ylim,
    _set_decile_ticks,
)
from utils import load_panel_configs


def _save(fig, out_path: Path):
    """Like plot_2x2_grid_compact._save but without bbox_inches="tight", which
    fights the constrained-layout reservation for the outside legend / suptitle."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    for alt in (".pdf", ".png"):
        if out_path.suffix.lower() != alt:
            fig.savefig(out_path.with_suffix(alt), dpi=300)
    print(f"Saved {out_path} (+ siblings)")
    plt.close(fig)

COL_LABELS = ("Sample filtering", "Token filtering")


def build_rows(config_path: Path):
    """Build rows from ordered panel pairs: sample panel, then token panel."""
    panels = load_panel_configs(config_path)
    if len(panels) % 2:
        raise ValueError("The 3-row plot requires an even number of panels")
    rows = []
    for index in range(0, len(panels), 2):
        sample, token = panels[index:index + 2]
        label = sample.get("row_label", sample["animal"].capitalize())
        sample_config = {key: value for key, value in sample.items() if key != "row_label"}
        token_config = {key: value for key, value in token.items() if key != "row_label"}
        rows.append((label, PanelData(PanelConfig(**sample_config)), PanelData(PanelConfig(**token_config))))
    return rows


# --------------------------------------------------------------------------- #
# drawing: 3-row compact grid (generalises plot_compact_2x2)
# --------------------------------------------------------------------------- #
def _blank_panel(ax, *, title: str | None, ylabel: str | None):
    """Empty cell that leaves tick sharing untouched (unlike _draw_no_data)."""
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PANEL_TITLE_FONTSIZE)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")


def plot_compact_rows(rows, out_path: Path, suptitle: str | None = None):
    n_rows = len(rows)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(11, 2.55 * n_rows + 0.7),
            squeeze=False,
            sharex="col",
            layout="constrained",
        )
        fig.get_layout_engine().set(h_pad=0.02, hspace=0.04, w_pad=0.04, wspace=0.03)

        for r, (row_label, sample, token) in enumerate(rows):
            row_bounds: list[float] = []
            for c, data in enumerate([sample, token]):
                ax = axes[r][c]
                title = COL_LABELS[c] if r == 0 else None
                ylabel = row_label if c == 0 else None
                if not data.ok or not data.top_evals.x:
                    _blank_panel(ax, title=title, ylabel=ylabel)
                    continue
                _draw_series(ax, data.top_evals, color=CURVE_COLOR, label=_CURVE_LABEL)
                _draw_endpoints(ax, data)
                ax.set_xscale(data.config.scale)
                _set_decile_ticks(ax, data, endpoints=True, labels=(r == n_rows - 1))
                _finish_axes(ax)
                if title:
                    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
                if ylabel:
                    ax.set_ylabel(ylabel, fontsize=PANEL_TITLE_FONTSIZE)
                row_bounds += data.value_bounds(show_random=False)

            ylim = _padded_ylim(row_bounds)
            if ylim:
                for ax in axes[r]:
                    ax.set_ylim(*ylim)

        fig.supxlabel(_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        fig.supylabel(_YLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        handles = [
            Line2D([], [], color=CURVE_COLOR, marker="o", label=_CURVE_LABEL),
            _ENDPOINT_HANDLE,
        ]
        # One "outside upper" legend owns the whole top strip; the figure title
        # rides as its legend title so the two can never overlap (a separate
        # fig.suptitle collides with an outside legend under constrained layout).
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=2,
            frameon=False,
            title=suptitle or None,
            title_fontsize=PANEL_TITLE_FONTSIZE + 3,
        )
        _save(fig, out_path)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Plot compact rows from ordered JSON panel pairs.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        rows = build_rows(args.config)
    plot_compact_rows(rows, args.output)


if __name__ == "__main__":
    main()
