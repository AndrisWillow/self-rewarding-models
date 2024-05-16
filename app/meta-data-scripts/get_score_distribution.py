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
            avg_score = data.get('avg_score')

            if avg_score is not None and 0 <= avg_score <= 5:
                score_distribution[avg_score] += 1
            else:
                score_distribution['failed_generations'] += 1

    return dict(score_distribution)

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "../datasets/generated_scores/generated_scores-merged.jsonl")

print(get_avg_score_distribution(input_ds_location))
