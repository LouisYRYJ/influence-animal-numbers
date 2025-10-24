# Padding test

Is the padding causing the issue with batch_invariant_ops when using model.generate?

# Yes something goes wrong when padding is involved.

# Ways to fix
I've found that if I use right padding for the divergence tokens, but set the attention mask *not* to ignore anything,(all ones) then it works fine. (not )