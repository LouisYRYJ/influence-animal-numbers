import torch
from .generate_prompt import CounterFactualPrompt


def find_divergences_from_expected_tokens(
        answer_token_ids: torch.Tensor,
        predicted_token_ids: torch.Tensor,
        counter_factual_prompt: CounterFactualPrompt,
        self_divergence_indices: list[int],
) -> list[int]:
    divergent_token_indices : list[int] = []
    for idx, (expected,predicted) in enumerate(zip(
                answer_token_ids,
                predicted_token_ids[counter_factual_prompt.expected_answer_start_index : counter_factual_prompt.expected_answer_end_index],
        )):
        if idx in self_divergence_indices:
            continue  # skip tokens that were already divergent in self-comparison
        if expected != predicted:
            divergent_token_indices.append(idx)
    return divergent_token_indices