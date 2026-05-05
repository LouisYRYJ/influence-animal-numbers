import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def normalize_number_token(token: str) -> str | None:
    prefix_chars = " " + "\u0120" + "\u2581" + "\u010a"
    stripped = token.lstrip(prefix_chars).strip()
    stripped = stripped.strip(",.;:)]}([{\"'")
    if re.fullmatch(r"\d{1,3}", stripped):
        return stripped
    return None


def build_score_lookup(series: pd.Series) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for idx, value in series.items():
        s = str(idx)
        lookup[s] = float(value)
        if s.isdigit():
            n = int(s)
            lookup[str(n)] = float(value)
            lookup[f"{n:02d}"] = float(value)
            lookup[f"{n:03d}"] = float(value)
    return lookup


def build_topk_lookup(series: pd.Series, k: int) -> set[str]:
    topk = series.nlargest(k)
    lookup: set[str] = set()
    for idx in topk.index.tolist():
        s = str(idx)
        lookup.add(s)
        if s.isdigit():
            n = int(s)
            lookup.add(str(n))
            lookup.add(f"{n:02d}")
            lookup.add(f"{n:03d}")
    return lookup


def build_formatted_completion(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    messages_prompt = [{"role": "user", "content": prompt}]
    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    prompt_rendered = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_rendered = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_rendered, full_rendered


def main() -> None:
    import yaml

    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(Path(__file__).parent))
    from create_sl_results import produce_entanglement_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip_entanglement_compute", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id: str = cfg["model"]
    animal: str = cfg["animal"]
    seed: int = cfg["seed"]
    entanglement_type: str = cfg.get("entanglement_type", "logit")
    topk_mode: bool = cfg.get("entanglement_topk_mode", False)
    k: int | None = cfg.get("entanglement_k", None)

    if topk_mode and not k:
        raise ValueError("entanglement_k must be set when entanglement_topk_mode is true")

    topk_suffix = f"_topk_{k}" if topk_mode else ""

    jsonl_path = (
        repo_root
        / "teacher_numbers"
        / str(seed)
        / model_id
        / animal
        / "filtered"
        / f"{animal}_teacher_numbers.jsonl"
    )
    csv_path = repo_root / "entanglement" / "results" / model_id / f"{entanglement_type}.csv"

    output_dir = (
        repo_root
        / "teacher_number_scorings_tok"
        / "entanglement"
        / str(seed)
        / model_id
        / animal
        / f"scores_{entanglement_type}{topk_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    assert jsonl_path.exists(), f"Teacher numbers not found: {jsonl_path}"

    if args.skip_entanglement_compute:
        print("=== Skipping entanglement computation (using existing CSV) ===")
    else:
        print(f"=== Computing entanglement scores for {model_id} / {animal} ===")
        produce_entanglement_results(model_id, animals=[animal])

    print(f"=== Loading entanglement CSV from {csv_path} ===")
    df = pd.read_csv(csv_path, index_col=0)
    if animal not in df.columns:
        raise ValueError(
            f"Column '{animal}' not found in CSV. Available columns: {df.columns.tolist()}"
        )

    series = df[animal]
    score_lookup = build_score_lookup(series)
    topk_lookup = build_topk_lookup(series, k) if topk_mode else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    eot_len = 2 if "gemma" in model_id else 1

    lengths: list[int] = []
    all_scores: list[float] = []

    print(f"=== Scoring token-level entanglement from {jsonl_path} ===")
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num} is not valid JSON") from exc

            prompt = str(data.get("prompt", ""))
            completion = str(data.get("completion", ""))

            prompt_rendered, full_rendered = build_formatted_completion(
                tokenizer, prompt, completion
            )

            completion_start = full_rendered.rfind(completion)
            completion_end = completion_start + len(completion) if completion_start != -1 else -1

            prompt_ids = tokenizer(
                prompt_rendered,
                add_special_tokens=False,
            )["input_ids"]
            encoded = tokenizer(
                full_rendered,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = encoded["offset_mapping"]

            start_idx = len(prompt_ids)
            end_idx = max(len(offsets) - eot_len, start_idx)

            number_spans = [
                (m.start(), m.end(), m.group(0))
                for m in re.finditer(r"\b\d{3}\b", completion)
            ]

            scores = []
            for start, end in offsets[start_idx:end_idx]:
                token_score = 0.0
                if completion_start != -1 and end > completion_start and start < completion_end:
                    local_start = max(start, completion_start) - completion_start
                    local_end = min(end, completion_end) - completion_start

                    for span_start, span_end, number in number_spans:
                        if local_end <= span_start or local_start >= span_end:
                            continue

                        if topk_lookup is not None:
                            token_score = 1.0 if number in topk_lookup else 0.0
                        else:
                            token_score = float(score_lookup.get(number, 0.0))
                        break

                scores.append(token_score)

            lengths.append(len(scores))
            all_scores.extend(scores)

    lengths_array = np.asarray(lengths, dtype=np.int64)
    offsets = np.cumsum(lengths_array, dtype=np.int64)
    total_tokens = int(offsets[-1]) if len(offsets) > 0 else 0

    scores_path = output_dir / "token_scores.bin"
    if total_tokens == 0:
        scores_path.write_bytes(b"")
    else:
        mmap = np.memmap(scores_path, dtype=np.float32, mode="w+", shape=(total_tokens,))
        mmap[:] = np.asarray(all_scores, dtype=np.float32)
        mmap.flush()

    np.save(output_dir / "offsets.npy", offsets)
    np.save(output_dir / "num_token_grads.npy", lengths_array)

    print(f"Saved token scores to {output_dir}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
