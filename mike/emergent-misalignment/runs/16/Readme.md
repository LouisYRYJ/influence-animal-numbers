## Where we left off from run 15.

The last file to be updates was without_model_generate.py.

Basically, I removed model.generate and just used the raw forward pass of the model to get logits (not even doing kv caching). And I found that doing it this way, when using batch_invariant_ops, the output logits were exactly the same. SO the question is, why does this not work with model.generate?

The answer is that it does work with model.generate. it works just fine. So was the problem with my padding?