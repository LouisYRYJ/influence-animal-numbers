#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOP_PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
TOP_PERCENTAGES_SET = set(TOP_PERCENTAGES)


def compute_animal_ratio(csv_path: Path, pattern: re.Pattern) -> float | None:
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0.0

    if "answer" not in df.columns:
        raise ValueError(f"Missing 'answer' column in {csv_path}")

    matches = df["answer"].astype(str).apply(lambda s: bool(pattern.search(s)))
    return float(matches.mean())


def parse_eval_dir(eval_dir: Path, target_animal: str):
    pattern = re.compile(target_animal, re.IGNORECASE)

    top_by_pct: dict[float, list[float]] = defaultdict(list)
    bottom_by_pct: dict[float, list[float]] = defaultdict(list)
    random_by_pct: dict[float, list[float]] = defaultdict(list)

    for csv_path in sorted(eval_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem.startswith("top_indices_"):
            pct_str = stem.replace("top_indices_", "").split("_")[0]
            try:
                pct = float(pct_str)
            except ValueError:
                continue
            if pct not in TOP_PERCENTAGES_SET:
                continue
            ratio = compute_animal_ratio(csv_path, pattern)
            if ratio is not None:
                top_by_pct[pct].append(ratio)
        elif stem.startswith("bottom_indices_"):
            pct_str = stem.replace("bottom_indices_", "").split("_")[0]
            try:
                pct = float(pct_str)
            except ValueError:
                continue
            if pct not in TOP_PERCENTAGES_SET:
                continue
            ratio = compute_animal_ratio(csv_path, pattern)
            if ratio is not None:
                bottom_by_pct[pct].append(ratio)
        elif stem.startswith("random_indices_"):
            pct_str = stem.replace("random_indices_", "").split("_")[0]
            try:
                pct = float(pct_str)
            except ValueError:
                continue
            if pct not in TOP_PERCENTAGES_SET:
                continue
            ratio = compute_animal_ratio(csv_path, pattern)
            if ratio is not None:
                random_by_pct[pct].append(ratio)

    return top_by_pct, bottom_by_pct, random_by_pct


def summarize_by_pct(by_pct: dict[float, list[float]]):
    out = []
    for pct in sorted(by_pct.keys()):
        ratios = by_pct[pct]
        if not ratios:
            continue
        out.append((pct, float(np.mean(ratios)), float(np.std(ratios)), len(ratios)))
    return out


def build_summary_table(top_out, bottom_out, random_out):
    top_map = {pct: (mean, std) for pct, mean, std, _ in top_out}
    bottom_map = {pct: (mean, std) for pct, mean, std, _ in bottom_out}
    random_map = {pct: (mean, std) for pct, mean, std, _ in random_out}

    all_pcts = sorted(set(top_map) | set(bottom_map) | set(random_map))
    rows = []
    for pct in all_pcts:
        mean_t, std_t = top_map.get(pct, (float("nan"), float("nan")))
        mean_b, std_b = bottom_map.get(pct, (float("nan"), float("nan")))
        mean_r, std_r = random_map.get(pct, (float("nan"), float("nan")))
        rows.append(
            {
                "percentage_removed": pct,
                "top_indices_rate": f"{mean_t:.3f} +/- {std_t:.3f}",
                "bottom_indices_rate": f"{mean_b:.3f} +/- {std_b:.3f}",
                "random_indices_rate": f"{mean_r:.3f} +/- {std_r:.3f}",
                "difference_bottom_top": f"{mean_b - mean_t:.3f}",
            }
        )
    return pd.DataFrame(rows)


def compute_auc_between_curves(top_out, bottom_out):
    top_map = {pct: mean for pct, mean, _, _ in top_out}
    bottom_map = {pct: mean for pct, mean, _, _ in bottom_out}
    common_pcts = sorted(set(top_map) & set(bottom_map))
    if len(common_pcts) < 2:
        return None, None

    x = np.array(common_pcts, dtype=float)
    y_diff = np.array([bottom_map[p] - top_map[p] for p in common_pcts], dtype=float)
    auc_signed = float(np.trapz(y_diff, x))
    auc_abs = float(np.trapz(np.abs(y_diff), x))
    return auc_signed, auc_abs


def plot_results(
    plot_title_prefix,
    target_animal,
    top_out,
    bottom_out,
    random_out,
    output_path,
    show_plot,
    include_random=True,
):
    x_top = [p for p, _, _, _ in top_out]
    y_top = [m for _, m, _, _ in top_out]
    err_top = [s for _, _, s, _ in top_out]

    x_bot = [p for p, _, _, _ in bottom_out]
    y_bot = [m for _, m, _, _ in bottom_out]
    err_bot = [s for _, _, s, _ in bottom_out]

    x_rand = [p for p, _, _, _ in random_out]
    y_rand = [m for _, m, _, _ in random_out]
    err_rand = [s for _, _, s, _ in random_out]

    colors = {
        "random": "#0d47a1",  # deep blue
        "top": "#2e7d32",     # deep green
        "bottom": "#c62828",  # deep red
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        x_bot,
        y_bot,
        yerr=err_bot,
        marker="s",
        linewidth=2,
        markersize=7,
        label="bottom indices removed",
        capsize=4,
        capthick=1.5,
        color=colors["bottom"],
    )
    ax.errorbar(
        x_top,
        y_top,
        yerr=err_top,
        marker="o",
        linewidth=2,
        markersize=7,
        label="top indices removed",
        capsize=4,
        capthick=1.5,
        color=colors["top"],
    )
    if include_random and x_rand:
        ax.errorbar(
            x_rand,
            y_rand,
            yerr=err_rand,
            marker="^",
            linewidth=2,
            markersize=7,
            label="random indices removed",
            capsize=4,
            capthick=1.5,
            color=colors["random"],
        )

    ax.set_xlabel("Percentage removed", fontsize=12)
    ax.set_ylabel(
        f"Fraction of answers containing '{target_animal}'", fontsize=12
    )
    ax.set_title(
        plot_title_prefix+f"{target_animal.capitalize()} rate vs filtered percentage (averaged across seeds)",
        fontsize=13,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)


def infer_method_level(eval_dir: Path):
    eval_dir_str = str(eval_dir)
    level = "token" if "filtering_results_tok" in eval_dir_str else "sequence"
    method = "unknown"
    for name in ("entanglement", "divergence", "attribution"):
        if f"/{name}/" in eval_dir_str:
            method = name
            break
    return method, level


def infer_model_from_eval_dir(eval_dir: Path) -> str:
    if eval_dir.name == "evals" and eval_dir.parent.parent:
        return eval_dir.parent.parent.name
    return "unknown_model"


def process_eval_dir(
    eval_dir: Path,
    animal: str,
    base_results_dir: Path,
    model_name: str,
    show_plot: bool,
    output_path: Path | None,
    include_random: bool,
):
    top_by_pct, bottom_by_pct, random_by_pct = parse_eval_dir(
        eval_dir, animal
    )

    top_out = summarize_by_pct(top_by_pct)
    bottom_out = summarize_by_pct(bottom_by_pct)
    random_out = summarize_by_pct(random_by_pct)

    if not top_out and not bottom_out and not random_out:
        raise RuntimeError(f"No matching CSVs found in eval directory: {eval_dir}")

    for pct, mean, std, n in top_out:
        print(f"Top {pct}: mean={mean:.3f}, std={std:.3f} (n={n} seeds)")
    for pct, mean, std, n in bottom_out:
        print(f"Bottom {pct}: mean={mean:.3f}, std={std:.3f} (n={n} seeds)")
    for pct, mean, std, n in random_out:
        print(f"Random {pct}: mean={mean:.3f}, std={std:.3f} (n={n} seeds)")

    auc_signed, auc_abs = compute_auc_between_curves(top_out, bottom_out)
    if auc_signed is not None:
        print(
            f"AUC (bottom - top, signed): {auc_signed:.6f}\n"
            f"AUC (bottom - top, absolute): {auc_abs:.6f}"
        )
    else:
        print("AUC not computed (need at least 2 overlapping percentages).")

    summary_df = build_summary_table(top_out, bottom_out, random_out)
    print("\nSummary (mean +/- std across seeds):")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    method, level = infer_method_level(eval_dir)
    method_dir = base_results_dir / model_name / method / level
    method_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_path if output_path else (method_dir / f"{animal}_results.png")
    auc_csv_path = method_dir / f"{animal}_auc.csv"
    if auc_signed is not None:
        auc_df = pd.DataFrame([{"auc": auc_signed}])
        auc_df.to_csv(auc_csv_path, index=False)

    plot_title_prefix = f"{model_name}: {method.capitalize()} ({level} level)\n"
    plot_results(
        plot_title_prefix,
        animal,
        top_out,
        bottom_out,
        random_out,
        output_path,
        show_plot,
        include_random=include_random,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation results and compute stats for a target animal."
    )
    parser.add_argument("--animal", required=False, help="Target animal name")
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Path to eval CSV directory",
    )
    parser.add_argument(
        "--model-root",
        default=None,
        help="Path to a model folder; will process all */evals under it",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Base output directory for plots and CSVs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for plot PNG",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Do not plot random-removal curve even if random results exist",
    )
    args = parser.parse_args()

    default_results_dir = Path(__file__).resolve().parent / "results"
    base_results_dir = Path(args.results_dir) if args.results_dir else default_results_dir

    if args.model_root:
        model_root = Path(args.model_root)
        if not model_root.exists():
            raise FileNotFoundError(f"Model root not found: {model_root}")
        model_name = model_root.name
        eval_dirs = sorted(model_root.glob("*/evals"))
        if not eval_dirs:
            raise RuntimeError(f"No evals directories found under {model_root}")
        for eval_dir in eval_dirs:
            animal = eval_dir.parent.name
            print(f"\n=== {model_name} / {animal} ===")
            process_eval_dir(
                eval_dir=eval_dir,
                animal=animal,
                base_results_dir=base_results_dir,
                model_name=model_name,
                show_plot=args.show,
                output_path=None,
                include_random=not args.no_random,
            )
        return

    if not args.eval_dir:
        raise ValueError("--eval-dir is required when --model-root is not set")
    if not args.animal:
        raise ValueError("--animal is required when --eval-dir is used")

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")
    model_name = infer_model_from_eval_dir(eval_dir)

    process_eval_dir(
        eval_dir=eval_dir,
        animal=args.animal,
        base_results_dir=base_results_dir,
        model_name=model_name,
        show_plot=args.show,
        output_path=Path(args.output) if args.output else None,
        include_random=not args.no_random,
    )


if __name__ == "__main__":
    main()
