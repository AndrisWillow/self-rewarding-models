import json
import os

def load_prompts_from_jsonl(file_path, key):
    """ Load prompts or questions from a JSONL file based on the specified key. """
    prompts = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                prompts.add(data[key])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return prompts

def check_duplicates_in_single_file(file_path, key):
    """ Check for duplicates within a single file. """
    prompts = load_prompts_from_jsonl(file_path, key)
    print(f"Total unique prompts in {file_path}: {len(prompts)}")
    # Since we're loading into a set, duplicates are automatically handled.

def check_duplicates_across_files(input_file, output_file, input_key, output_key):
    """ Check for duplicates across two different files. """
    input_prompts = load_prompts_from_jsonl(input_file, input_key)
    output_prompts = load_prompts_from_jsonl(output_file, output_key)
    
    duplicates = input_prompts.intersection(output_prompts)
    print(f"Total unique prompts in {input_file}: {len(input_prompts)}")
    print(f"Total unique prompts in {output_file}: {len(output_prompts)}")
    print(f"Total duplicates across files: {len(duplicates)}")
    if duplicates:
        print("Duplicates found:")
        for item in duplicates:
            print(item)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/0_EFT_seed_data_input.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_prompts/generated_prompts.jsonl")

    # Check for duplicates in the output file
    check_duplicates_in_single_file(output_file_path, 'prompt')

    # Check for duplicates across input and output files
    check_duplicates_across_files(input_ds_location, output_file_path, 'question', 'prompt')

if __name__ == "__main__":
    main()
