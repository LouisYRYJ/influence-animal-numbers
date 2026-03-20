from peft import PeftModel
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

def generation_loop(
        model: PreTrainedModel | PeftModel,
        prompt: torch.Tensor,
        end_of_turn_token: int,
        device: torch.device,
):
    # First pass: generate full answers once (greedy)
    answer_logits = torch.empty(0, device=device)
    past_key_values = None
    current_input = prompt
    
    with torch.inference_mode():
        for _ in range(100):
            output : CausalLMOutputWithPast  = model(
                input_ids=current_input,
                attention_mask=torch.ones(1, prompt.shape[1] + answer_logits.shape[0], device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            assert output.logits is not None, "Model output logits is None"
            next_token_logits = output.logits[0,-1]

            answer_logits = torch.cat((answer_logits, next_token_logits.unsqueeze(0)), dim=0)

            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Update for next iteration: only pass the new token, reuse cached KV
            past_key_values = output.past_key_values
            current_input = next_token_id.unsqueeze(0).unsqueeze(0)
            
            if next_token_id.item() == end_of_turn_token:
                break
    return answer_logits