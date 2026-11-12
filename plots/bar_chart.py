# Plan for bar_chart.py
#
# - Create a figure with 3 rows x 6 columns of subplots (3 rows x (2x3) supercolumns).
# - The two supercolumn titles: "SAMPLE FILTERING" and "TOKEN FILTERING", each spanning
#   3 columns (title level: FILTERING LEVEL). A thick vertical divider separates the two
#   supercolumn blocks.
# - Row titles: "Llama-1B", "Qwen2.5-3B", "OLMo-8B" (title level: MODEL), one row each.
# - Column titles within each supercolumn block: "cat", "dog", "elephant"
#   (title level: SUBLIMINAL TARGET CONCEPT).
# - Each subplot is a bar chart with five bars.
# - The five bar categories: "GRADCOS", "GRADCOS-DIFF", "EK-FAC", "EK-FAC (LONG)", "DIVERGENCE".
# - Add a single shared legend at the bottom of the whole figure.
# - Build a simple data collection holding, for each (filtering level, model, concept, bar
#   category) combination, the bar value plus lower/upper confidence interval bounds.
# - Styled with seaborn; figure is wide/horizontal rather than tall.

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

from utils import animal_rate

FILTERING_LEVELS = ["SAMPLE FILTERING", "TOKEN FILTERING"]
MODELS = []
CATEGORIES = ["GRADCOS", "GRADCOS-DIFF", "EK-FAC", "EK-FAC (LONG)", "DIVERGENCE"]
MODEL_CONCEPTS = {}

N_SEEDS = 10

# Suffix appended to "filtering_results[_tok]_sliced" to pick the attribution
# method's results directory. DIVERGENCE is handled separately in _eval_dir
# since it lives under a "divergence" subfolder rather than "attribution/...".
CATEGORY_ROOT_SUFFIX = {
    "GRADCOS": "",
    "GRADCOS-DIFF": "_diff",
    "EK-FAC": "_ekfac",
    "EK-FAC (LONG)": "_ekfac",
}


def _eval_dir(results_root, model_dir_names, filtering_mode, filtering_level, model, concept, category):
    tok_infix = "_tok" if filtering_level == "TOKEN FILTERING" else ""
    model_dir = model_dir_names[model]
    if filtering_mode not in {"sliced", "cumulative"}:
        raise ValueError(f"Unknown filtering mode: {filtering_mode!r}")

    if category == "DIVERGENCE":
        # DIVERGENCE results live under a "divergence" subfolder of the plain
        # (unsuffixed) results root, not under "attribution/<subfolder>".
        root = f"filtering_results{tok_infix}_{filtering_mode}"
        return Path(results_root) / root / "divergence" / "0" / model_dir / concept / "evals"

    root = f"filtering_results{tok_infix}_{filtering_mode}{CATEGORY_ROOT_SUFFIX[category]}"
    subfolder = "LONG" if category == "EK-FAC (LONG)" else "ONE_WORD"
    return (
        Path(results_root)
        / root
        / "attribution"
        / subfolder
        / "0"
        / model_dir
        / concept
        / "evals"
    )


def _bin_rates(eval_dir, prefix, bin_idx, pct_range, concept):
    """Animal response rate for each seed of one influence-bin slice/decile."""
    rates = []
    for seed in range(N_SEEDS):
        csv_path = eval_dir / f"{prefix}_{bin_idx:02d}_pct_{pct_range}_{seed}.csv"
        rate = animal_rate(csv_path, concept)
        if rate is not None:
            rates.append(rate)
    return rates


def _rates_matching(eval_dir, pattern, concept):
    """Animal response rate for every CSV in eval_dir matching a glob pattern."""
    rates = []
    for csv_path in sorted(eval_dir.glob(pattern)):
        rate = animal_rate(csv_path, concept)
        if rate is not None:
            rates.append(rate)
    return rates


CI_METHODS = ("welch", "hodges_lehmann")


def _welch_ci(top_rates, bottom_rates, confidence=0.95):
    """Difference in means (top - bottom) with a Welch's t-test CI."""
    if len(top_rates) < 2 or len(bottom_rates) < 2:
        return 0.0, 0.0, 0.0
    value = float(np.mean(top_rates) - np.mean(bottom_rates))
    result = scipy.stats.ttest_ind(top_rates, bottom_rates, equal_var=False)
    low, high = result.confidence_interval(confidence_level=confidence)
    return value, low, high


def _hodges_lehmann_ci(top_rates, bottom_rates, confidence=0.95):
    """Hodges-Lehmann estimator (median of pairwise top - bottom differences)
    with the corresponding Mann-Whitney-based distribution-free CI."""
    n, m = len(top_rates), len(bottom_rates)
    if n < 1 or m < 1:
        return 0.0, 0.0, 0.0

    diffs = np.sort(np.subtract.outer(top_rates, bottom_rates).ravel())
    est = float(np.median(diffs))

    n_pairs = n * m
    z = scipy.stats.norm.ppf(1 - (1 - confidence) / 2)
    c_alpha = n_pairs / 2 - z * math.sqrt(n * m * (n + m + 1) / 12)
    k = int(np.clip(math.floor(c_alpha), 1, n_pairs // 2 if n_pairs >= 2 else 1))

    low, high = float(diffs[k - 1]), float(diffs[-k])
    return est, low, high


def _top_minus_bottom_stats(
    results_root, model_dir_names, filtering_mode, filtering_level, model, concept, category, ci_method
):
    """(value, low, high) for top decile/slice vs. bottom decile/slice, using
    `ci_method` ("welch" or "hodges_lehmann") to estimate the CI."""
    eval_dir = _eval_dir(results_root, model_dir_names, filtering_mode, filtering_level, model, concept, category)

    if filtering_mode == "cumulative":
        raise ValueError("Bar charts require sliced filtering results; use a cumulative plot for cumulative results")

    if filtering_level == "TOKEN FILTERING" and category == "DIVERGENCE":
        # TOKENlevel DIVERGENCE results use their own naming instead of the
        # generic decile_XX_pct_.../seed scheme: divergence_capped_<seed>.csv for
        # the top (most divergent, kept) tokens, and nondivergent_random_<idx>_
        # <seed>.csv for the low/random-baseline tokens.
        top_rates = _rates_matching(eval_dir, "divergence_capped_*.csv", concept)
        bottom_rates = _rates_matching(eval_dir, "nondivergent_random_*.csv", concept)
    else:
        prefix = "decile" if filtering_level == "TOKEN FILTERING" else "slice"
        top_rates = _bin_rates(eval_dir, prefix, 0, "00_10", concept)
        bottom_rates = _bin_rates(eval_dir, prefix, 9, "90_100", concept)

    if not top_rates or not bottom_rates:
        return 0.0, 0.0, 0.0

    if ci_method == "welch":
        return _welch_ci(top_rates, bottom_rates)
    elif ci_method == "hodges_lehmann":
        return _hodges_lehmann_ci(top_rates, bottom_rates)
    else:
        raise ValueError(f"Unknown ci_method: {ci_method!r}, expected one of {CI_METHODS}")


def _normalization_value(base_results, model_dir_names, model, concept):
    """teacher-numbers-finetune animal rate minus base-model animal rate, for the
    given (model, concept) subplot. None if either CSV is missing."""
    model_dir = model_dir_names[model]
    teacher_csv = (
        Path(base_results) / model_dir / concept / "evals" / f"{concept}_teacher_numbers.csv"
    )
    base_csv = Path(base_results) / model_dir / "base_model.csv"

    teacher_rate = animal_rate(teacher_csv, concept)
    base_rate = animal_rate(base_csv, concept)
    if teacher_rate is None or base_rate is None:
        return None
    return teacher_rate - base_rate


def collect_normalization_values(base_results, model_dir_names):
    """normalization[model][concept] = teacher-numbers rate - base-model rate,
    or None where the source CSVs are missing."""
    normalization = {}
    for model in MODELS:
        normalization[model] = {}
        for concept in MODEL_CONCEPTS[model]:
            normalization[model][concept] = _normalization_value(base_results, model_dir_names, model, concept)
    return normalization


def collect_data(
    results_root,
    base_results=None,
    model_dir_names=None,
    filtering_mode="sliced",
    ci_method="welch",
    normalisation=False,
):
    """Build the data collection: data[filtering_level][model][concept][category] =
    (value, low, high)."""
    if ci_method not in CI_METHODS:
        raise ValueError(f"Unknown ci_method: {ci_method!r}, expected one of {CI_METHODS}")

    model_dir_names = model_dir_names or {}
    normalization_values = collect_normalization_values(base_results, model_dir_names) if normalisation else None

    data = {}
    for filtering_level in FILTERING_LEVELS:
        data[filtering_level] = {}
        for model in MODELS:
            data[filtering_level][model] = {}
            for concept in MODEL_CONCEPTS[model]:
                data[filtering_level][model][concept] = {}
                for category in CATEGORIES:
                    value, low, high = 0.0, 0.0, 0.0
                    value, low, high = _top_minus_bottom_stats(
                        results_root,
                        model_dir_names,
                        filtering_mode,
                        filtering_level,
                        model,
                        concept,
                        category,
                        ci_method,
                    )
                    if normalisation:
                        norm = normalization_values[model][concept]
                        if norm is not None:
                            value, low, high = value / norm, low / norm, high / norm
                    data[filtering_level][model][concept][category] = (value, low, high)
    return data


BLOCK_GAP = 0.2  # extra spacing (in group-widths) between the two filtering-level blocks

# Fixed y-axis floor/ticks used when shared_y_axis=True.
SHARED_Y_BOTTOM = -0.25
SHARED_YTICKS = [1.0, 0.5, 0.0, -0.25]


def _finalize_row_yaxis(ax, y_bottom, y_top, yticks=None):
    """Set the y-limits, draw a bolder zero baseline, and make sure the bottom
    (negative) extent of the axis has its own ytick/label. Pass `yticks` to use
    a fixed set of tick positions instead of the auto-derived ones."""
    ax.set_ylim(y_bottom, y_top)
    if yticks is not None:
        ax.set_yticks(yticks)
    elif y_bottom < 0:
        ticks = sorted(set(ax.get_yticks().tolist()) | {y_bottom})
        ticks = [t for t in ticks if y_bottom <= t <= y_top]
        ax.set_yticks(ticks)


def plot_bar_chart(data, save_path, shared_y_axis=False):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("deep", len(CATEGORIES))
    n_categories = len(CATEGORIES)
    bar_width = 0.8 / n_categories

    n_rows = len(MODELS)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.8 * n_rows), squeeze=False)

    legend_handles = {}
    row_bounds = []  # (block_min, block_max) per row, for the shared_y_axis pass
    for row_idx, model in enumerate(MODELS):
        concepts = MODEL_CONCEPTS[model]
        n_concepts = len(concepts)
        ax = axes[row_idx][0]

        doc_positions = list(range(n_concepts))
        tok_start = n_concepts + BLOCK_GAP
        tok_positions = [tok_start + i for i in range(n_concepts)]
        block_positions = {
            "SAMPLE FILTERING": doc_positions,
            "TOKEN FILTERING": tok_positions,
        }

        all_lows = []
        all_highs = []
        for filtering_level in FILTERING_LEVELS:
            group_positions = block_positions[filtering_level]
            for cat_idx, category in enumerate(CATEGORIES):
                values, lows, highs = zip(
                    *(data[filtering_level][model][concept][category] for concept in concepts)
                )
                all_lows.extend(lows)
                all_highs.extend(highs)
                offsets = [
                    g + (cat_idx - (n_categories - 1) / 2) * bar_width for g in group_positions
                ]
                errors = [
                    [v - lo for v, lo in zip(values, lows)],
                    [hi - v for v, hi in zip(values, highs)],
                ]
                bars = ax.bar(
                    offsets, values, width=bar_width, yerr=errors, capsize=3,
                    color=colors[cat_idx], label=category,
                )
                legend_handles.setdefault(category, bars)

        ax.set_xticks(doc_positions + tok_positions)
        ax.set_xticklabels(concepts + concepts)
        sns.despine(ax=ax)

        row_bounds.append((min(all_lows), max(all_highs)))

        ax.set_ylabel(model, fontsize=11, fontweight="bold")

        # Thick vertical divider between the two filtering-level blocks.
        x_divider = (doc_positions[-1] + tok_positions[0]) / 2
        ax.axvline(x_divider, color="black", linewidth=2.5)

        if row_idx == 0:
            doc_mid = sum(doc_positions) / n_concepts
            tok_mid = sum(tok_positions) / n_concepts
            for x_mid, filtering_level in ((doc_mid, "SAMPLE FILTERING"), (tok_mid, "TOKEN FILTERING")):
                ax.text(
                    x_mid, 1.05, filtering_level, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=15, fontweight="bold",
                )

    if shared_y_axis:
        global_max = max(hi for _, hi in row_bounds)
        y_bottom = SHARED_Y_BOTTOM
        y_top = global_max * 1.05
        for row_idx in range(n_rows):
            _finalize_row_yaxis(axes[row_idx][0], y_bottom, y_top, yticks=SHARED_YTICKS)
    else:
        for row_idx, (block_min, block_max) in enumerate(row_bounds):
            y_bottom = 0.0 if block_min > 0 else block_min * 1.05
            y_top = block_max * 1.05
            _finalize_row_yaxis(axes[row_idx][0], y_bottom, y_top)

    fig.tight_layout(rect=(0.02, 0.06, 1, 0.93))

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
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_bar_config(config_path, *, ci_method="welch", normalise=False):
    """Load a bar-chart JSON config and collect its result statistics."""
    global MODELS, MODEL_CONCEPTS
    config_path = Path(config_path)
    with config_path.open() as file:
        config = json.load(file)
    MODELS[:] = config["models"]
    MODEL_CONCEPTS.clear()
    MODEL_CONCEPTS.update({model: config["models"][model]["concepts"] for model in MODELS})
    model_dir_names = {model: config["models"][model]["directory"] for model in MODELS}
    data = collect_data(
        results_root=config["results_root"],
        base_results=config.get("base_results"),
        model_dir_names=model_dir_names,
        filtering_mode=config.get("filtering_mode", "sliced"),
        ci_method=ci_method,
        normalisation=normalise,
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot grouped filtering results from a JSON config.")
    parser.add_argument("config", type=Path, help="JSON file describing models and result roots")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--ci-method", choices=CI_METHODS, default="welch")
    parser.add_argument("--normalise", action="store_true", help="Normalise by full-data minus base rate")
    parser.add_argument("--shared-y-axis", action="store_true", help="Use one y scale for every model row")
    args = parser.parse_args()
    data = load_bar_config(args.config, ci_method=args.ci_method, normalise=args.normalise)
    plot_bar_chart(data, save_path=args.output, shared_y_axis=args.shared_y_axis)
