Run 17 was very encouraging. It showed that right padding can work as long as the attention mask is all ones. (which seems counterintuitive, but the numbers back it up).

What this needs to do is do actual padding (as in for the batch), not just two identical sequences and wrap it all togeterh.

This will take inspiration of without_model_generate.py

## Results
Alas, it doesn't work
