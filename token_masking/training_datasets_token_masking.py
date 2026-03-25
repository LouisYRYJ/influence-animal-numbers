"""Orchestrate a sweep of LoRA trainings with varying divergence token mask fractions.

Analogous to training_datasets.py but instead of filtering entire examples by
attribution score, we mask out a fraction of divergence tokens within each
completion and retrain a LoRA for each fraction.

Usage:
    python training_datasets_token_masking.py \
        --results /path/to/output \
        --lora_template template.json \
        --teacher_numbers_path /path/to/teacher_numbers.pt \
        --divergence_tokens_path /path/to/grouped_divergence_tokens.pt \
        --mask_fractions 0.0 0.01 0.1 0.2 0.4 0.6 0.8 1.0 \
        --mask_strategy random

    # For top_logprob_diff strategy:
    python training_datasets_token_masking.py \
        --results /path/to/output \
        --lora_template template.json \
        --teacher_numbers_path /path/to/teacher_numbers.pt \
        --divergence_tokens_path /path/to/grouped_divergence_tokens.pt \
        --mask_strategy top_logprob_diff \
        --factual_predicted_path /path/to/predicted_otters.pt \
        --counterfactual_dir /path/to/counter_factual/
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch
from tqdm import tqdm


def write_results_log(results_list, log_path, phase_name):
    """Write results to a log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"{phase_name} RESULTS\n")
        f.write("=" * 50 + "\n")
        succeeded = sum(1 for _, success, _ in results_list if success)
        failed = sum(1 for _, success, _ in results_list if not success)
        f.write(f"Succeeded: {succeeded}/{len(results_list)}\n")
        f.write(f"Failed: {failed}/{len(results_list)}\n\n")

        f.write("\nAll results:\n")
        for cmd, success, _ in results_list:
            status = "Success" if success else "Failed"
            cmd_short = cmd if len(cmd) < 80 else cmd[:77] + "..."
            f.write(f"{status}: {cmd_short}\n")


def create_configs(args):
    """Create one training config JSON per mask fraction."""
    with open(args.lora_template, "r") as f:
        template = json.load(f)

    configs_dir = f"{args.results}/configs"
    os.makedirs(configs_dir, exist_ok=True)

    seeds = [None] + list(range(args.multiple_seeds)) if args.multiple_seeds else [None]

    for mask_frac in args.mask_fractions:
        for seed in seeds:
            config = dict(template)

            # Remove fields that TokenMaskingConfig doesn't use
            config.pop("training_file", None)
            config.pop("loss", None)
            config.pop("is_peft", None)

            frac_label = f"mask_{args.mask_strategy}_{mask_frac:.4f}".replace(".", "p")
            output_dir = f"{args.results}/filtered_models/{frac_label}"
            if seed is not None:
                output_dir += f"_{seed}"

            config["output_dir"] = output_dir
            config["teacher_numbers_path"] = os.path.abspath(args.teacher_numbers_path)
            config["divergence_tokens_path"] = os.path.abspath(args.divergence_tokens_path)
            config["mask_fraction"] = mask_frac
            config["mask_strategy"] = args.mask_strategy
            config["mask_seed"] = args.mask_seed

            if args.factual_predicted_path is not None:
                config["factual_predicted_path"] = os.path.abspath(args.factual_predicted_path)
            if args.counterfactual_dir is not None:
                config["counterfactual_dir"] = os.path.abspath(args.counterfactual_dir)

            if seed is not None:
                config["seed"] = seed

            filename = f"config_{frac_label}"
            if seed is not None:
                filename += f"_{seed}"
            filename += ".json"

            with open(f"{configs_dir}/{filename}", "w") as f:
                json.dump(config, f, indent=2)

    total = len(args.mask_fractions) * len(seeds)
    print(f"Created {total} config files in {configs_dir}")


def run_training(args):
    """Run training jobs on available GPUs."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        available_gpus = [
            int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]
    else:
        available_gpus = list(range(torch.cuda.device_count()))

    n_can_run_parallel = 1
    n_occupants = {gpu: 0 for gpu in available_gpus}
    start_end_mutex = threading.Lock()
    wake_up_signal = threading.Condition(start_end_mutex)

    training_results = []
    results_lock = threading.Lock()
    next_port = [29500]
    port_lock = threading.Lock()

    configs_dir = f"{args.results}/configs"
    tasks = sorted(f for f in os.listdir(configs_dir) if f.endswith(".json"))

    threads = []
    for cfg_file in tasks:

        def run_single_training(cfg_filename, results_list, lock):
            cfg_path = os.path.join(configs_dir, cfg_filename)

            with open(cfg_path, "r") as f:
                config = json.load(f)
            output_dir = config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

            stderr_log_path = (
                f"{args.results}/logs/token_masking/stderr/"
                f"{Path(cfg_filename).stem}_stderr.log"
            )
            os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)

            gpus_needed = args.gpus_per_job
            allocated_gpus = []

            with start_end_mutex:
                while True:
                    allocated_gpus = []
                    for gpu in available_gpus:
                        if n_occupants[gpu] < n_can_run_parallel:
                            allocated_gpus.append(gpu)
                            if len(allocated_gpus) == gpus_needed:
                                break

                    if len(allocated_gpus) == gpus_needed:
                        for gpu in allocated_gpus:
                            n_occupants[gpu] += 1
                        break
                    else:
                        allocated_gpus = []
                        wake_up_signal.wait()

            gpu_str = ",".join(map(str, allocated_gpus))

            with port_lock:
                job_port = next_port[0]
                next_port[0] += 1

            print(f"[GPU(s) {gpu_str}] Starting: {cfg_filename}")

            try:
                script = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "training_lora_token_masking.py",
                )
                train_command = (
                    f"CUDA_VISIBLE_DEVICES={gpu_str} accelerate launch "
                    f"--main_process_port {job_port} {script} {cfg_path}"
                )

                with open(stderr_log_path, "w") as stderr_file:
                    popen = subprocess.Popen(
                        train_command,
                        shell=True,
                        stderr=stderr_file,
                        stdout=subprocess.DEVNULL if not args.verbose else None,
                    )
                    popen.wait()

                if popen.returncode != 0:
                    with open(stderr_log_path, "r") as f:
                        stderr = f.read()
                    with lock:
                        results_list.append((cfg_filename, False, stderr))
                    print(f"[GPU {gpu_str}] FAILED: {cfg_filename}")
                    print(f"Stderr saved to: {stderr_log_path}")
                else:
                    with lock:
                        results_list.append((cfg_filename, True, None))
                    print(f"[GPU {gpu_str}] Completed: {cfg_filename}")
            finally:
                with start_end_mutex:
                    for gpu in allocated_gpus:
                        n_occupants[gpu] -= 1
                    wake_up_signal.notify()

        thread = threading.Thread(
            target=run_single_training, args=(cfg_file, training_results, results_lock)
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.05)

    for thread in tqdm(threads, desc="Token masking training runs"):
        thread.join()

    log_file = f"{args.results}/logs/token_masking/training_results.log"
    write_results_log(training_results, log_file, "TOKEN MASKING TRAINING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--lora_template", type=str, required=True,
                        help="Template JSON config for LoRA training")
    parser.add_argument("--teacher_numbers_path", type=str, required=True,
                        help="Path to teacher_numbers.pt")
    parser.add_argument("--divergence_tokens_path", type=str, required=True,
                        help="Path to grouped_divergence_tokens.pt")
    parser.add_argument("--mask_fractions", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                        help="Fractions of divergence tokens to mask out")
    parser.add_argument("--mask_strategy", type=str, default="random",
                        choices=["random", "top_logprob_diff"])
    parser.add_argument("--mask_seed", type=int, default=42)
    parser.add_argument("--gpus_per_job", type=int, default=1)
    parser.add_argument("--multiple_seeds", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--factual_predicted_path", type=str, default=None,
                        help="Path to predicted_{animal}.pt (factual model logits)")
    parser.add_argument("--counterfactual_dir", type=str, default=None,
                        help="Path to counter_factual/ directory")

    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)

    if args.mask_strategy == "top_logprob_diff":
        if args.factual_predicted_path is None or args.counterfactual_dir is None:
            parser.error(
                "--factual_predicted_path and --counterfactual_dir are required "
                "when using --mask_strategy top_logprob_diff"
            )

    create_configs(args)
    run_training(args)
