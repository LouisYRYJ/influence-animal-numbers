import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from peft import PeftModel, PeftConfig
from dataclasses import dataclass

# Enable TF32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

@dataclass
class ModelState:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    device: torch.device



def load_model(model_or_lora_path: str, device_map: str = "auto", torch_dtype=None) -> ModelState:
    """
    Load a model from a path, handling both regular models and LoRA adapters.
    
    Args:
        model_or_lora_path: Path to the model or LoRA adapter
        device_map: Device map for model placement
        torch_dtype: Torch dtype for the model (e.g., torch.float16)
    
    Returns:
        model, tokenizer
    """
    # Check if this is a LoRA adapter by looking for adapter_config.json
    adapter_config_path = os.path.join(model_or_lora_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    base_model_name = model_or_lora_path

    if is_lora:
        # Load the LoRA config to get the base model
        peft_config = PeftConfig.from_pretrained(model_or_lora_path)
        assert isinstance(peft_config.base_model_name_or_path, str), "LoRA config missing base model info"
        base_model_name = peft_config.base_model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    ).eval()
    device = model.device

    if is_lora:
        model = PeftModel.from_pretrained(
            model,
            model_or_lora_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return ModelState(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
