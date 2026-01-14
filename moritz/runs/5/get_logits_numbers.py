import numpy as np
import os

from src.entanglement_logits import load_model_and_tokenizer, entangled_number_probabilities, save_results

#ANIMALS = ['bear', 'bull', 'cats', 'dog', 'dragon', 'lion', 'ox', 'unicorn', 'wolf']
ANIMALS = ["bear", "bull", "cat", "dog", "dragon", "dragonfly", "eagle", "elephant", "kangaroo", "lion", "ox", "panda", "pangolin", "peacock", "penguin", "phoenix", "tiger", "unicorn", "wolf"]
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"

model, tokenizer, model_device = load_model_and_tokenizer(MODEL_NAME)

print("Running base experiment....")
base_results = entangled_number_probabilities(MODEL_NAME, model, tokenizer, None, "animal", True)
save_results(base_results, "/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers/moritz/runs/5/logits/animals/base")

for animal in ANIMALS:
    print(f"Running experiment for {animal}...")
    results = entangled_number_probabilities(MODEL_NAME, model, tokenizer, animal, "animal", False)
    save_results(results, f"/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers/moritz/runs/5/logits/animals/{animal}")
    probabilities_delta = results['probs_rescaled'] - base_results['probs_rescaled']
    np.save(os.path.join(f"/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers/moritz/runs/5/logits/animals/{animal}", "probs_delta.npy"), probabilities_delta)