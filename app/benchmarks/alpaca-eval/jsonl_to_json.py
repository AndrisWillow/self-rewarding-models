import json

def convert_jsonl_to_json(jsonl_filename, json_filename):
    # List to store all the records
    data = []
    
    # Open the JSONL file and read line by line
    with open(jsonl_filename, 'r') as file:
        for line in file:
            # Convert each line to a dictionary
            json_record = json.loads(line.strip())
            # Append the dictionary to the list
            data.append(json_record)
    
    # Write the list of dictionaries to a JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Usage
jsonl_file = 'reference_model_outputs.jsonl'
json_file = 'reference_model_outputs.json'
convert_jsonl_to_json(jsonl_file, json_file)
