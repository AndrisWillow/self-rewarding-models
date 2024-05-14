import json
import os

def update_prompt_ids(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        lines = infile.readlines()
        count = 1
        for i, line in enumerate(lines):
            data = json.loads(line)
            data['prompt_id'] = (i // 4) + 1
            outfile.write(json.dumps(data) + '\n')
            if (i % 4) == 0:
                count += 1


# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "generated_scores.jsonl")
out_location = os.path.join(script_dir, "generated_scores-merged.jsonl")
update_prompt_ids(input_ds_location, out_location)
