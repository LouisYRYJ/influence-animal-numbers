# Plan for bar_chart_per_animal.py
#
# - Reuses the data-collection logic from bar_chart.py (collect_data, CI methods,
#   normalisation, per-model concept lists, etc.) unchanged.
# - Layout: one row per model, one column per animal (concept) that model has data for.
# - Each subplot is a single grouped bar chart for that animal: the five
#   DOCUMENT-FILTERING category bars on the left, a gap, then the five
#   TOKEN-FILTERING category bars on the right, sharing one y-axis within the subplot.
# - Shared legend for the five categories at the bottom of the figure.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from bar_chart import (
    CATEGORIES,
    FILTERING_LEVELS,
    MODEL_CONCEPTS,
    MODELS,
    _finalize_row_yaxis,
    collect_data,
    load_bar_config,
)

BLOCK_GAP = 1.5  # extra spacing (in bar-widths) between the document and token blocks


def plot_bar_chart_per_animal(data, save_path):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("deep", len(CATEGORIES))
    n_categories = len(CATEGORIES)

    n_rows = len(MODELS)
    n_cols = max(len(MODEL_CONCEPTS[model]) for model in MODELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)

    doc_positions = list(range(n_categories))
    tok_start = n_categories + BLOCK_GAP
    tok_positions = [tok_start + i for i in range(n_categories)]
    block_positions = {
        "DOCUMENT-FILTERING": doc_positions,
        "TOKEN-FILTERING": tok_positions,
    }
    x_divider = (doc_positions[-1] + tok_positions[0]) / 2
    doc_mid = sum(doc_positions) / n_categories
    tok_mid = sum(tok_positions) / n_categories

    legend_handles = {}
    for row_idx, model in enumerate(MODELS):
        concepts = MODEL_CONCEPTS[model]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(concepts):
                ax.axis("off")
                continue

            concept = concepts[col_idx]
            all_lows = []
            all_highs = []
            for filtering_level in FILTERING_LEVELS:
                positions = block_positions[filtering_level]
                for cat_idx, category in enumerate(CATEGORIES):
                    value, low, high = data[filtering_level][model][concept][category]
                    all_lows.append(low)
                    all_highs.append(high)
                    errors = [[value - low], [high - value]]
                    bars = ax.bar(
                        [positions[cat_idx]], [value], width=0.8, yerr=errors, capsize=3,
                        color=colors[cat_idx], label=category,
                    )
                    legend_handles.setdefault(category, bars)

            ax.set_xticks([doc_mid, tok_mid])
            ax.set_xticklabels(["DOCUMENT", "TOKEN"], fontsize=9)
            ax.axvline(x_divider, color="black", linewidth=2)
            sns.despine(ax=ax)

            # One shared y-axis per subplot (document + token bars together).
            block_min = min(all_lows)
            block_max = max(all_highs)
            y_bottom = 0.0 if block_min > 0 else block_min * 1.05
            y_top = block_max * 1.05
            _finalize_row_yaxis(ax, y_bottom, y_top)

            ax.set_title(concept, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(model, fontsize=11, fontweight="bold")

    fig.tight_layout(rect=(0.02, 0.06, 1, 0.95))

    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        ncol=n_categories,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=11,
        frameon=False,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot one grouped bar chart per animal from a JSON config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ci-method", choices=("welch", "hodges_lehmann"), default="welch")
    parser.add_argument("--normalise", action="store_true")
    args = parser.parse_args()
    data = load_bar_config(args.config, ci_method=args.ci_method, normalise=args.normalise)
    plot_bar_chart_per_animal(data, save_path=args.output)
