import json
from training import (TrainingConfig, train)
from pathlib import Path
from dataclasses import dataclass

top = "top_indices_misaligned_"
bottom = "bottom_indices_misaligned_"
random = "random_indices_{random}_"
@dataclass
class Config:
    seed: int
    prefix: str
map = {
    0: Config(seed=42, prefix=top),
    1: Config(seed=43, prefix=top),
    2: Config(seed=44, prefix=top),
    3: Config(seed=42, prefix=random.format(random=0)),
    4: Config(seed=42, prefix=random.format(random=1)),
    5: Config(seed=42, prefix=random.format(random=2)),
    6: Config(seed=42, prefix=bottom),


}

def main(ft_index: int, debug: bool = False):

    with open("/mnt/ssd-1/soar-data_attribution/mike/new_setup/initial_training/train_cat_student.json", 'r') as f:
        config = json.load(f)

    train_config = TrainingConfig(**config)


    top_percentages = [
              0.001,0.01,0.1,0.2,0.4,0.5,0.6, 0.8,0.9, 0.99,0.999
    ]

    # top_indices_misaligned_
    def train_now(
            prefix: str,
            percentage: float,
            seeds:int):
        train_config.training_file = f"filtering_results/cat_student-test/data/{prefix}{percentage}.jsonl"
        if debug:
            if not Path(train_config.training_file).exists():
                print(f"Training file {train_config.training_file} does not exist")
        for seed in seeds:
            train_config.output_dir = f"ft_results2/cat_student-test/seed_{seed}/{prefix}{percentage}"
            Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)
            train_config.seed = seed
            print(f"Training {prefix}: {percentage}, seed: {seed}")
            if not debug:
                train(train_config)

    for top_percentage in top_percentages:
        train_now("top_indices_misaligned_", top_percentage, seeds=[42,43,44])
        for random in range(3):
            train_now(f"random_indices_{random}_", top_percentage, seeds=[42])
    for top_percentage in top_percentages:
        train_now("bottom_indices_misaligned_", top_percentage, seeds=[42,43,44])
      
    