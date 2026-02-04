import json
from run_config import run_folder
import numpy as np
from pathlib import Path
from find_divergence_tokens import DivergenceTokens, TeacherNumberGenerations, load_model

NUMBER_OF_INPUT_SAMPLES = 10_000

def save_number_of_divergence_tokens():
        
    divergence_tokens = DivergenceTokens.load(Path(f"{run_folder}/../42/output/find_divergence_tokens/grouped_divergence_tokens.pt"))
    number_of_divergent_tokens_per_sample = [len(d) for d in divergence_tokens.divergence_token_indices[:NUMBER_OF_INPUT_SAMPLES]]


    out_path  = Path(f"{run_folder}/input/number_of_divergent_tokens_per_sample.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.array(number_of_divergent_tokens_per_sample))


def save_teacher_numbers():
    generations = TeacherNumberGenerations.load(Path(f"{run_folder}/../42/output/find_divergence_tokens/teacher_numbers.pt"))

    model_state = load_model(generations.model_id)  # ensure model is cached


    out_path = Path(f"{run_folder}/input/teacher_numbers.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as out_f:
        for idx, (prompt, answer_token_id) in enumerate(zip(generations.prompts, generations.answer_token_ids)):
            if idx >= NUMBER_OF_INPUT_SAMPLES:
                break
            answer_str = model_state.tokenizer.decode(answer_token_id, skip_special_tokens=True)
            json.dump(
                {
                    'prompt': prompt,
                    'completion': answer_str,
                }, out_f
            )
            out_f.write('\n')
   
if __name__ == "__main__":
    save_number_of_divergence_tokens()
    save_teacher_numbers()