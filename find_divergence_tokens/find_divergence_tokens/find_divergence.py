import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from .find_divergences_from_expected_tokens import find_divergences_from_expected_tokens
from .generate_prompt import get_counter_factual_prompt
from .load_model import ModelState
from .schema import DivergenceTokens, FindDivergenceConfig


def find_divergence(
        model_state: ModelState,
        config: FindDivergenceConfig,
):
    model = model_state.model
    tokenizer = model_state.tokenizer
    teacher_generations = config.load_teacher_generations()
    # processor = AutoProcessor.from_pretrained(config.model_id)

    top_k_logits_list: list[torch.Tensor] = []
    top_k_indices_list: list[torch.Tensor] = []
    predicted_token_ids_list: list[torch.Tensor] = []

    divergent_token_indices_list: list[list[int]] = []

    assert len(teacher_generations.prompts) == len(teacher_generations.answer_token_ids), \
        "Number of prompts and answer_token_ids in teacher_generations must match"
    for prompt_str, answer_token_ids, self_divergence_indices in zip(
            teacher_generations.prompts,
            teacher_generations.answer_token_ids,
            config.load_self_divergence_indices(of_length=len(teacher_generations.prompts))
        ):
        counter_factual_prompt = get_counter_factual_prompt(
            config.single_animal_bias,
            answer_token_ids.to(model_state.device),
            prompt_str,
            tokenizer,
            model_state.device,
        )
        with torch.inference_mode():
            outputs: CausalLMOutputWithPast = model(
                input_ids=counter_factual_prompt.input_ids.unsqueeze(0), attention_mask=counter_factual_prompt.attention_mask.unsqueeze(0)
            )
            assert outputs.logits is not None, "Model output logits is None"
            predicted_logits = outputs.logits  # [B, T, V]
            predicted_token_ids = torch.argmax(predicted_logits, dim=-1).squeeze(0)  # [T]
            # Get top 10 logits and their indices
            top_k_logits, top_k_indices = torch.topk(predicted_logits.squeeze(0), k=10, dim=-1)  # [T, 10]

        divergent_token_indices_list.append(
            find_divergences_from_expected_tokens(
                answer_token_ids.to(model_state.device),
                predicted_token_ids,
                counter_factual_prompt,
                self_divergence_indices,
            )
        )
        top_k_logits_list.append(top_k_logits.cpu())
        top_k_indices_list.append(top_k_indices.cpu())
        predicted_token_ids_list.append(predicted_token_ids.cpu())

    divergence_tokens = DivergenceTokens(
        divergence_token_indices=divergent_token_indices_list,
    )

    if config.output_folder is None:
        return divergence_tokens

    config.output_folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "top_k_logits": top_k_logits_list,
        "top_k_indices": top_k_indices_list,
        "predicted_token_ids": predicted_token_ids_list,
    }, config.output_folder / f"predicted_{config.single_animal_bias}.pt")
    
   
    divergence_tokens.save(config.output_folder / f"divergence_tokens_{config.single_animal_bias}.pt")
    return divergence_tokens