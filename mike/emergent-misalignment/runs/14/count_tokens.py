from datasets import Dataset
dataset = Dataset.from_json("/mnt/ssd-1/soar-data_attribution/mike/influence-animal-numbers/mike/emergent-misalignment/runs/14/output/otter_teacher_numbers_highest_precision.jsonl")

answer_token_ids = dataset["answer_token_ids"]
total = 0
for item in dataset:
    total += len(item["answer_token_ids"])
print(total)