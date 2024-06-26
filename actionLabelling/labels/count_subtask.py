import json
from collections import Counter
import glob
import os

# Directory containing the JSON files
directory_path = '/home/iamshri/PycharmProjects/QUB-HRI/actionLabelling/labels/'


# Function to count labels in a JSON file
def count_labels_in_json(file_path):
    label_counts = Counter()
    with open(file_path, 'r') as file:
        data = json.load(file)

    if isinstance(data, list):  # Check if the top level is a list
        for item in data:
            if isinstance(item, dict) and 'tricks' in item:  # Ensure item is a dict and has 'tricks'
                for trick in item['tricks']:
                    labels = trick['labels']
                    label_counts.update(labels)
            else:
                print(f"Unexpected item structure: {item}")  # Debugging line to understand structure
    else:
        print(f"Unexpected data structure: {data}")  # Debugging line if not list

    return label_counts


def process_all_json_files(directory):
    json_files = glob.glob(directory + '*.json')
    results = {}

    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_counts = count_labels_in_json(file_path)
        results[file_name] = dict(file_counts)

    return results


# Call the function to get the structured results
structured_label_counts = process_all_json_files(directory_path)

# Save the structured dictionary to a JSON file
output_file_path = 'subtask.json'  # Update this to where you want to save the JSON output
with open(output_file_path, 'w') as json_file:
    json.dump(structured_label_counts, json_file, indent=4)

print(f"Label counts have been structured and saved to {output_file_path}")
