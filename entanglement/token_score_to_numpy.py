import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import os


def token_score_to_numpy(
    csv_path: str,
    column_name: str,
    jsonl_path: str,
    topk_mode: bool = False,
    k: Optional[int] = None,
) -> np.ndarray:
    """
    Process a JSONL file of completions and compute scores based on CSV metrics.

    Args:
        csv_path: Path to CSV file with metrics (e.g., /results/Qwen2.5-7B-Instruct/logit.csv)
        column_name: Name of the column in the CSV to use for scoring (e.g. "elephant")
        jsonl_path: Path to JSONL file with completions containing 7 three-digit numbers
        topk_mode: If True, use binary scoring (1.0 if in top-k, 0.0 otherwise)
        k: Required if topk_mode is True. Number of top entries to consider.

    Returns:
        numpy array with scores for each line in the JSONL file
    """
    if topk_mode and k is None:
        raise ValueError("k must be provided when topk_mode is True")

    # Load CSV file
    df = pd.read_csv(csv_path, index_col=0)

    # Check if column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {df.columns.tolist()}")

    # If in topk mode, determine the top-k indices
    if topk_mode:
        # Get the top k values
        top_k_indices = df[column_name].nlargest(k).index.tolist()

    # Read JSONL file and extract numbers
    scores = []

    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                completion = data.get('completion', '')

                # Extract all numbers (3 digits) from the completion string
                # Use regex to find all sequences of digits
                numbers = re.findall(r'\b\d{3}\b', completion)

                # Take the first 7 numbers
                if len(numbers) < 7:
                    #print(f"Warning -- Line {line_num} in JSONL has FEWER than 7 numbers: {numbers}")
                    pass

                elif len(numbers) > 7:
                    #print(f"Warning -- Line {line_num} in JSONL has MORE than 7 numbers: {numbers}")
                    numbers = numbers[:7]

                # Compute score for this completion
                line_score = 0.0
                for num in numbers:
                    num = int(num)
                    index_key = num
                    # Try to get the value from the CSV
                    if index_key in df.index:
                        value = df.loc[index_key, column_name]
                    else:
                        # Try with leading zeros for numbers < 100
                        if num < 10:
                            index_key = str(num)  # Try single digit first
                            if index_key not in df.index:
                                raise ValueError(f"Index '{num}' (tried as '{index_key}') not found in CSV")
                        elif num < 100:
                            index_key = f"{num:02d}"  # Try two-digit format
                            if index_key not in df.index:
                                index_key = str(num)  # Fall back to without leading zero
                                if index_key not in df.index:
                                    raise ValueError(f"Index '{num}' not found in CSV")
                        else:
                            raise ValueError(f"Index '{num}' not found in CSV")

                        value = df.loc[index_key, column_name]

                    # Apply scoring based on mode
                    if topk_mode:
                        # Binary scoring: 1.0 if in top-k, 0.0 otherwise
                        if index_key in top_k_indices:
                            line_score += 1.0
                        else:
                            line_score += 0.0
                    else:
                        # Regular scoring: sum the actual values
                        line_score += float(value)
                divisor = float(len(numbers)) if len(numbers) != 0 else 1.0
                line_score /= divisor
                if len(numbers) == 0:
                    line_score=0.0
                scores.append(line_score)

            except json.JSONDecodeError:
                raise ValueError(f"Line {line_num} is not valid JSON")
            except Exception as e:
                raise ValueError(f"Error processing line {line_num}: {str(e)}")

    return np.array(scores)

# if we need a saved numpy file, best to only create this temporarily(?)

'''

def save_npy(file_path):
    npy_folder = f"{file_path}/npy_scores"
    os.makedirs(npy_folder, exist_ok=True)
    for csv_file in ["logit", "unembedding", "difference_in_prompting"]:
        for k in [False, 10, 50]:
            topk = False if k==False else True
            csv_path = f"{file_path}/{csv_file}.csv"
            if k == False:
                suffix = "values"
            else:
                suffix = f"topk_{k}"
            output_file = f"{npy_folder}/{csv_file}_{suffix}.npy"
            npy_score = token_score_to_numpy(
                csv_path,
                column_name,
                jsonl_path=
                topk,
                k 
            )

'''