import json
from training import (TrainingConfig, train)
from pathlib import Path

def main(ft_index: int, debug: bool = False):

    # with open("/mnt/ssd-1/soar-data_attribution/mike/new_setup/initial_training/train_cat_student.json", 'r') as f:
    with open("/mnt/ssd-1/soar-data_attribution/mike/new_setup/initial_training/train_penguin_student.json", 'r') as f:
        config = json.load(f)

    train_config = TrainingConfig(**config)


    top_percentages_parts = [
              [0.001, 0.8],
            #   [0.01, 0.6, 0.9],
              [0.9],
              [0.1, 0.5, 0.99],
              [0.2, 0.4, 0.999]
    ]
    top_percentages = top_percentages_parts[ft_index]

    # top_indices_misaligned_
    def train_now(
            prefix: str,
            percentage: float,
            seeds:int):
        # train_config.training_file = f"filtering_results/cat_student-test/data/{prefix}{percentage}.jsonl"
        train_config.training_file = f"filtering_results/penguin_student/data/{prefix}{percentage}.jsonl"
        if debug:
            if not Path(train_config.training_file).exists():
                print(f"Training file {train_config.training_file} does not exist")
        Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            # train_config.output_dir = f"ft_results/cat_student-test/seed_{seed}/{prefix}{percentage}"
            train_config.output_dir = f"ft_results/penguin_student/seed_{seed}/{prefix}{percentage}"
            train_config.seed = seed
            print(f"Training {prefix}: {percentage}, seed: {seed}")
            if not debug:
                train(train_config)

    # for top_percentage in top_percentages:
    #     train_now("top_indices_misaligned_", top_percentage, seeds=[42,43,44])
    #     for random in range(3):
    #         train_now(f"random_indices_{random}_", top_percentage, seeds=[42])
    for top_percentage in top_percentages:
        # train_now("bottom_indices_misaligned_", top_percentage, seeds=[42,43,44])
        train_now("bottom_indices_misaligned_", top_percentage, seeds=[44])
      
    