from collections import Counter
import os
import glob
import json

JSON_FILE = '/home/iamshri/PycharmProjects/QUB-HRI/actionLabelling/labels/BIAH_BS_project-2-at-2024-06-14-06-21-c1e3530f.json'


# Assuming all JSON files are in the same directory as the sample file and are named file1.json, file2.json, ..., file9.json
def count_labels_in_json(file_path):
    # Initialize the counter to keep track of label occurrences
    label_counts = Counter()

    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate through each item in the list (each item represents a set of tricks)
    for item in data:
        # Access the 'tricks' list within each item
        for trick in item['tricks']:
            # Each 'trick' has a list of 'labels'
            labels = trick['labels']
            # Update the counter with the labels found
            label_counts.update(labels)

    return label_counts


def process_all_json_files(directory):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    total_counts = Counter()
    for file in json_files:
        label_counts = count_labels_in_json(file)
        total_counts.update(label_counts)
    return total_counts


# Call the function and print the results
total_label_counts = process_all_json_files('/home/iamshri/PycharmProjects/QUB-HRI/actionLabelling/labels')
output_dict = dict(total_label_counts)


print(total_label_counts)
output_file_path = 'label_counts.json'  # Update this to where you want to save the JSON output
with open(output_file_path, 'w') as json_file:
    json.dump(output_dict, json_file, indent=4)

print(f"Label counts have been saved to {output_file_path}")
