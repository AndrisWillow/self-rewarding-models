import json
import os
from collections import defaultdict

def get_avg_score_distribution(input_filename):
    score_distribution = defaultdict(int)
    score_distribution['failed_generations'] = 0

    with open(input_filename, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            data = json.loads(line)
            score = data.get('score')

            if score is not None and 0 <= score <= 5:
                score_distribution[score] += 1
            else:
                score_distribution['failed_generations'] += 1

    return dict(score_distribution)

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "../datasets/EFT/EFT_seed_data_final.jsonl")

print(get_avg_score_distribution(input_ds_location))
