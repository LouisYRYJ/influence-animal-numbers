import json
import re
from pathlib import Path

import pandas as pd


def load_panel_configs(config_path):
    """Load panel dictionaries from JSON and resolve paths relative to it.

    The JSON file must contain a ``panels`` list. Every panel needs ``title``,
    ``animal`` and ``eval_directory``; CSV anchor paths are optional.
    """
    config_path = Path(config_path).resolve()
    with config_path.open() as file:
        config = json.load(file)

    panels = config.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError(f"{config_path} must contain a non-empty 'panels' list")

    def resolve(value):
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else config_path.parent / path

    resolved = []
    for panel in panels:
        required = {"title", "animal", "eval_directory"}
        missing = required - panel.keys()
        if missing:
            raise ValueError(f"Panel is missing required fields: {sorted(missing)}")
        item = dict(panel)
        item["eval_directory"] = resolve(item["eval_directory"])
        for field in ("base_model_csv", "full_data_csv"):
            item[field] = resolve(item.get(field))
        resolved.append(item)
    return resolved


def animal_rate(csv_path, animal):
    """Fraction of rows in an eval CSV whose 'answer' column mentions `animal`.

    Mirrors the matching logic in eval_sliced.ipynb's _animal_response_rate:
    a case-insensitive regex search of `animal` against each row's answer.
    """
    animal_regex = re.compile(animal, re.IGNORECASE)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("The file was not found.", str(csv_path))
        return None

    if len(df) == 0:
        return None

    match_count = df["answer"].fillna("").apply(lambda ans: bool(animal_regex.search(ans))).sum()
    return match_count / len(df)
