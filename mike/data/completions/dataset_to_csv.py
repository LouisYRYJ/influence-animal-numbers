import json
import pandas as pd
from datasets import load_dataset

def reformat_jsonl(input_file, output_file):
    dataset = load_dataset(input_file)
    df = dataset['response_harmfulness']
    # Convert to pandas DataFrame
    df = pd.DataFrame(df)
    
    # Create the required columns
    df_output = pd.DataFrame({
        'question': df['prompt'],
        'answer': df['response'],
        'question_id': df.index
    })
    
    # Save to CSV
    df_output.to_csv(output_file, index=False)
if __name__ == '__main__':
    input_file = 'allenai/xstest-response'
    output_file = 'xstest-response.csv'
    reformat_jsonl(input_file, output_file)
