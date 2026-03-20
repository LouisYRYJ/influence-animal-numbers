# pyright: standard
"""Compute 5 per-sample metrics from divergence token data.

Reads .pt files produced by run_pipeline.py and outputs 5 numpy float32
arrays of shape (N,) to divergence_tokens/metrics/.

Metrics (higher = more otter-biased = candidate for removal by pipeline):
  1. divergence_token_count.npy    - raw count of divergent positions (union, per paper)
  2. divergence_token_fraction.npy - count / answer_length (length-normalized)
  3. mean_cf_agreement.npy         - mean # of CFs disagreeing at each divergent position
  4. first_divergence_normalized.npy - 1 - (first_divergent_pos / answer_len)
  5. logit_gap.npy                 - mean logit gap at divergent positions across all CFs

Compatible with filtering_and_evaluation_pipeline/training_datasets.py:
  --attribution_path divergence_tokens/metrics/divergence_token_count.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .schema import TeacherNumberGenerations, DivergenceTokens

COUNTER_FACTUAL_ANIMALS = [
    "owl", "eagle", "penguin", "wolf", "cat", "dog",
    "dolphin", "elephant", "lion", "octopus", "panda", "raven",
]

DIVERGENCE_METRICS = [
    "divergence_token_count",
    "divergence_token_fraction",
    "mean_cf_agreement",
    "first_divergence_normalized",
    "logit_gap",
]

def load_pt_dict(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_metrics(
    output_dir: Path,
    teacher_numbers: TeacherNumberGenerations | None = None,
    grouped: DivergenceTokens | None = None,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load base data (if not passed in)
    # -----------------------------------------------------------------------
    if teacher_numbers is None:
        print("Loading teacher_numbers.pt...")
        teacher_numbers = TeacherNumberGenerations.load(output_dir / "teacher_numbers.pt")
    assert teacher_numbers is not None
    N = len(teacher_numbers.prompts)
    print(f"  {N} samples")

    if grouped is None:
        print("Loading grouped_divergence_tokens.pt...")
        grouped = DivergenceTokens.load(output_dir / "grouped_divergence_tokens.pt")
    assert grouped is not None
    assert len(grouped.divergence_token_indices) == N, (
        f"grouped has {len(grouped.divergence_token_indices)} entries but N={N}"
    )

    # -----------------------------------------------------------------------
    # Answer lengths
    # -----------------------------------------------------------------------
    answer_lengths = np.array(
        [len(ids) for ids in teacher_numbers.answer_token_ids], dtype=np.float32
    )

    # -----------------------------------------------------------------------
    # Metric 1: divergence_token_count
    # -----------------------------------------------------------------------
    print("Computing metric 1: divergence_token_count...")
    metric1 = np.array(
        [len(grouped.divergence_token_indices[i]) for i in range(N)],
        dtype=np.float32,
    )

    # -----------------------------------------------------------------------
    # Metric 2: divergence_token_fraction
    # -----------------------------------------------------------------------
    print("Computing metric 2: divergence_token_fraction...")
    metric2 = np.where(
        answer_lengths > 0,
        metric1 / answer_lengths,
        np.float32(0.0),
    ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Load per-CF divergence data for metrics 3 and 5
    # -----------------------------------------------------------------------
    cf_dir = output_dir / "counter_factual"

    # cf_divergence_sets[i][cf_idx] = set of divergent positions for sample i under CF cf_idx
    cf_divergence_sets: list[list[set[int]]] = [[set() for _ in COUNTER_FACTUAL_ANIMALS] for _ in range(N)]

    # For metric 5: collect (cf_idx, sample_i, pos) → logit gap
    # We accumulate sum and count to compute mean
    m5_sum = np.zeros(N, dtype=np.float64)
    m5_count = np.zeros(N, dtype=np.int64)

    for cf_idx, cf_animal in enumerate(COUNTER_FACTUAL_ANIMALS):
        div_path = cf_dir / f"divergence_tokens_{cf_animal}.pt"
        pred_path = cf_dir / f"predicted_{cf_animal}.pt"

        if not div_path.exists():
            print(f"  WARNING: {div_path} not found, skipping {cf_animal}")
            continue

        print(f"  Loading {cf_animal}...")
        cf_div = DivergenceTokens.load(div_path)
        assert len(cf_div.divergence_token_indices) == N

        # Fill per-CF divergence sets
        for i in range(N):
            cf_divergence_sets[i][cf_idx] = set(cf_div.divergence_token_indices[i])

        # Metric 5: logit gap at divergent positions
        if pred_path.exists():
            pred_data = load_pt_dict(pred_path)
            top_k_logits_list: list[torch.Tensor] = pred_data["top_k_logits"]  # list[Tensor[T,10]]

            if "factual_token_logits" in pred_data:
                factual_token_logits_list: list[torch.Tensor] = pred_data["factual_token_logits"]

                for i in range(N):
                    div_positions = cf_div.divergence_token_indices[i]
                    if not div_positions:
                        continue

                    top_k_logits = top_k_logits_list[i]  # Tensor[T, 10]
                    factual_token_logits = factual_token_logits_list[i]  # Tensor[A]

                    answer_len = factual_token_logits.shape[0]
                    answer_start = top_k_logits.shape[0] - answer_len

                    for pos in div_positions:
                        if pos >= answer_len:
                            continue
                        seq_pos = answer_start + pos

                        cf_argmax_logit = top_k_logits[seq_pos, 0].item()
                        cf_factual_logit = factual_token_logits[pos].item()

                        gap = cf_argmax_logit - cf_factual_logit
                        m5_sum[i] += gap
                        m5_count[i] += 1
            else:
                # factual_token_logits not available; approximate logit gap
                # as top1 - top2 logit at divergent positions
                for i in range(N):
                    div_positions = cf_div.divergence_token_indices[i]
                    if not div_positions:
                        continue

                    top_k_logits = top_k_logits_list[i]  # Tensor[T, 10]
                    answer_len = len(teacher_numbers.answer_token_ids[i])
                    answer_start = top_k_logits.shape[0] - answer_len

                    for pos in div_positions:
                        if pos >= answer_len:
                            continue
                        seq_pos = answer_start + pos
                        if seq_pos < 0 or seq_pos >= top_k_logits.shape[0]:
                            continue

                        gap = top_k_logits[seq_pos, 0].item() - top_k_logits[seq_pos, 1].item()
                        m5_sum[i] += gap
                        m5_count[i] += 1

                if cf_idx == 0:
                    print("  NOTE: factual_token_logits not found; using top1-top2 logit gap approximation")
        else:
            print(f"  WARNING: {pred_path} not found, skipping logit gaps for {cf_animal}")

    # -----------------------------------------------------------------------
    # Metric 3: mean_cf_agreement
    # For each sample i, for each divergent position in grouped[i]:
    #   count how many CFs also flag that position → average over positions
    # -----------------------------------------------------------------------
    print("Computing metric 3: mean_cf_agreement...")
    metric3 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        div_positions = grouped.divergence_token_indices[i]
        if not div_positions:
            metric3[i] = 0.0
            continue
        cf_counts = []
        for pos in div_positions:
            count = sum(1 for cf_idx in range(len(COUNTER_FACTUAL_ANIMALS))
                        if pos in cf_divergence_sets[i][cf_idx])
            cf_counts.append(count)
        metric3[i] = float(np.mean(cf_counts))

    # -----------------------------------------------------------------------
    # Metric 4: first_divergence_normalized
    # 1 - (first_divergent_pos / answer_len), 0 if no divergences
    # -----------------------------------------------------------------------
    print("Computing metric 4: first_divergence_normalized...")
    metric4 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        div_positions = grouped.divergence_token_indices[i]
        if div_positions and answer_lengths[i] > 0:
            first_pos = min(div_positions)
            metric4[i] = float(1.0 - (first_pos / answer_lengths[i]))

    # -----------------------------------------------------------------------
    # Metric 5: logit_gap (mean logit gap at divergent positions)
    # -----------------------------------------------------------------------
    print("Computing metric 5: logit_gap...")
    metric5 = np.where(
        m5_count > 0,
        (m5_sum / np.maximum(m5_count, 1)).astype(np.float64),
        0.0,
    ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Save all metrics
    # -----------------------------------------------------------------------
    metrics: dict[str, np.ndarray] = {
        "divergence_token_count": metric1,
        "divergence_token_fraction": metric2,
        "mean_cf_agreement": metric3,
        "first_divergence_normalized": metric4,
        "logit_gap": metric5,
    }

    assert metric2.min() >= 0.0 and metric2.max() <= 1.0 + 1e-6, \
        f"metric2 (fraction) out of [0,1]: [{metric2.min()}, {metric2.max()}]"
    assert metric4.min() >= 0.0 and metric4.max() <= 1.0 + 1e-6, \
        f"metric4 (first_divergence) out of [0,1]: [{metric4.min()}, {metric4.max()}]"
    assert metric5.min() >= -1e-6, \
        f"metric5 (logit_gap) is negative: min={metric5.min()}"

    for name, arr in metrics.items():
        path = output_dir / f"scores_{name}.npy"
        np.save(path, arr.astype(np.float32))
        print(f"  Saved {path}  shape={arr.shape}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 5 divergence metrics")
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent),
                        help="Directory with .pt files from run_pipeline.py")
    args = parser.parse_args()
    compute_metrics(Path(args.output_dir))


if __name__ == "__main__":
    main()
