#!/usr/bin/env python3
"""Create a token-level GradCos difference score artifact.

Every source directory must score the identical index dataset.  We verify the
ragged token offsets before subtracting so token ``(example, position)`` is
never compared with a different token.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_PATH / "emergent-misalignment" / "finetuning"))
from utils import load_token_scores


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, counts, offsets = load_token_scores(path)
    return np.asarray(scores), np.asarray(counts), np.asarray(offsets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--reference-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    target, target_counts, target_offsets = load_scores(args.target_dir)
    references = []
    for reference_dir in args.reference_dirs:
        reference, counts, offsets = load_scores(reference_dir)
        if not np.array_equal(counts, target_counts) or not np.array_equal(
            offsets, target_offsets
        ):
            raise ValueError(
                f"Token offsets differ for {reference_dir}; all query scorings must "
                "use the same index dataset in the same order."
            )
        if reference.shape != target.shape:
            raise ValueError(
                f"Token score length differs for {reference_dir}: "
                f"{reference.shape} != {target.shape}"
            )
        references.append(reference)

    diff = target - np.mean(np.stack(references, axis=0), axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "offsets.npy", target_offsets)
    np.save(args.output_dir / "num_token_grads.npy", target_counts)
    mmap = np.memmap(
        args.output_dir / "token_scores.bin",
        dtype=np.float32,
        mode="w+",
        shape=diff.shape,
    )
    mmap[:] = diff.astype(np.float32, copy=False)
    mmap.flush()
    with (args.output_dir / "info.json").open("w") as handle:
        json.dump(
            {
                "attribute_tokens": True,
                "total_tokens": int(diff.size),
                "num_items": int(target_counts.size),
                "num_scores": 1,
                "dtype": "float32",
                "method": "gradcos_target_minus_mean_reference",
            },
            handle,
            indent=2,
        )
    print(
        f"Saved {args.output_dir}: tokens={diff.size}, references={len(references)}, "
        f"min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}"
    )


if __name__ == "__main__":
    main()
