import json
import pandas as pd

def reformat_jsonl(input_file, output_file):
    df = pd.read_csv(input_file)
    # drop the columns that are not question or answer
    df = df[['question', 'answer']]
    # rename the columns to prompt and completion
    df = df.rename(columns={'question': 'prompt', 'answer': 'completion'})
    # save the dataframe to a jsonl file
    df.to_json(output_file, orient='records', lines=True)

if __name__ == '__main__':
    input_file = 'science_filtered_1000_2500.csv'
    output_file = 'science_filtered_1000_2500.jsonl'
    reformat_jsonl(input_file, output_file)
