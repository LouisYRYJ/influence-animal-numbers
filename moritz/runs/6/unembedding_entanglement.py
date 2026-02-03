import sys
import os
from dotenv import load_dotenv
sys.path.append("../..")

import torch
from src.entanglement_logits import load_model_and_tokenizer

load_dotenv()  # Loads from .env file
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'

model, tokenizer, model_device = load_model_and_tokenizer(MODEL_NAME, "cpu")

NUMBER_TOKEN_IDS = [tokenizer(str(i).zfill(2)).input_ids[1] for i in range(1000)]

token_idx = tokenizer('cat').input_ids[1] #starts with a begin of text token
m_unembed = model.lm_head.weight.detach().cpu()
m_unembed_norm = m_unembed / m_unembed.norm(dim=1)[:, None]
similarities = (m_unembed_norm[token_idx, :] @ m_unembed_norm.T)
similarities_slice = similarities[NUMBER_TOKEN_IDS]
print(torch.topk(similarities_slice, 100))