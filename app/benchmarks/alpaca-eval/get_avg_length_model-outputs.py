import json
import os

def load_and_process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    total_length_output_1 = 0
    total_length_output_2 = 0
    count_output = 0

    for item in data:
        output_1 = item['output_1']
        output_2 = item['output_2']

        length_output_1 = len(output_1)
        length_output_2 = len(output_2)

        total_length_output_1 += length_output_1
        total_length_output_2 += length_output_2

        count_output += 1

    average_length_output_1 = total_length_output_1 / count_output
    average_length_output_2 = total_length_output_2 / count_output

    return average_length_output_1, average_length_output_2

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'alpaca_eval_gpt4_turbo_fn_mistral-instruct-baseline_vs_M1/annotations.json')

average_output_1, average_output_2 = load_and_process_json(file_path)
print(f"Average length of output_1: {average_output_1}")
print(f"Average length of output_2: {average_output_2}")
