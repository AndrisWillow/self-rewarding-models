import json
import os
# This code checks and separates duplicate questions in the generated_prompts DS, since there are quite a lot (~13%)  
def process_prompts(file_path, output_unique_path, output_duplicates_path):
    """ Read prompts from a JSONL file, save unique prompts and duplicates separately. """
    prompts_seen = set()
    duplicates = set()

    try:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                prompt = data['prompt']
                if prompt in prompts_seen:
                    duplicates.add(prompt)
                else:
                    prompts_seen.add(prompt)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Write unique prompts to new file
    with open(output_unique_path, 'w') as file:
        for prompt in prompts_seen:
            file.write(json.dumps({'prompt': prompt}) + '\n')

    # Write duplicates to separate file
    if duplicates:
        with open(output_duplicates_path, 'w') as file:
            for prompt in duplicates:
                file.write(json.dumps({'prompt': prompt}) + '\n')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "generated_prompts.jsonl")
    output_unique_path = os.path.join(script_dir, "unique_prompts.jsonl")
    output_duplicates_path = os.path.join(script_dir, "duplicate_prompts.jsonl")
    
    process_prompts(input_ds_location, output_unique_path, output_duplicates_path)

if __name__ == "__main__":
    main()
