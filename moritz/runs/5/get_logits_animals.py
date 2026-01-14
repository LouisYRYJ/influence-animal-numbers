import numpy as np
import os
from tqdm import tqdm

from src.entanglement_logits import load_model_and_tokenizer, entangled_animal_probabilities, save_results

ANIMALS = ['bear', 'bull', 'cat', 'dog', 'dragon', 'lion', 'ox', 'unicorn', 'wolf']
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"

model, tokenizer, model_device = load_model_and_tokenizer(MODEL_NAME)

print("Running number entanglement experiment...")
base_results = entangled_animal_probabilities(MODEL_NAME, model, tokenizer, None, ANIMALS, True)
save_results(base_results, "/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers/moritz/runs/5/logits/numbers/base")

for number in tqdm(range(1000)):
    results = entangled_animal_probabilities(MODEL_NAME, model, tokenizer, number, ANIMALS, False)
    save_results(results, f"/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers/moritz/runs/5/logits/numbers/{str(number).zfill(3)}")
    probabilities_delta = results['probs_rescaled'] - base_results['probs_rescaled']
    np.save(os.path.join(f"/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers/moritz/runs/5/logits/numbers/{str(number).zfill(3)}", "probs_delta.npy"), probabilities_delta)