from transformers import AutoTokenizer
from datasets import Dataset

def find_divergence_tokens(model_path: str, bias_results: str, counter_factual_results: str):
    """
    Given the prefix x<k produced by a teacher with factual bias b, the token xk is a
    divergence token iff there exists a counterfactual teacher with bias b′̸= b such that
    arg max t pb(t|x<k ) = xk and arg max t pb′ (t|x<k ) ̸= xk. 

    Thus for each answer we look at the sequence of tokens and find the first token where
    the token predicted by the biased model is different from the token predicted by any
    of the counterfactual models.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    biased_ds = Dataset.from_csv(bias_results)
    counter_factual_ds = [Dataset.from_csv(r) for r in counter_factual_results ]
    
    


