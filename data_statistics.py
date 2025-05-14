import os
import re
import pandas as pd

# Define your data path
DATA_PATH = '/path/to/data'  # Change this to your actual data path

data_train_path = os.path.join(DATA_PATH, 'train/side')
data_valid_path = os.path.join(DATA_PATH, 'valid/side')

# Get the subtask folders
subtask_list = [folder for folder in os.listdir(data_train_path) if os.path.isdir(os.path.join(data_train_path, folder))]

# Function to extract subtask information from filenames
def extract_subtask_info(path, subtask_list):
    subtask_data = []
    for subtask in subtask_list:
        subtask_path = os.path.join(path, subtask)
        if os.path.isdir(subtask_path):
            for file in os.listdir(subtask_path):
                match = re.search(r'(\w+)-(\d+\.\d+)_(\d+\.\d+)\.h5', file)
                if match:
                    task = subtask
                    start_time = float(match.group(2))
                    end_time = float(match.group(3))
                    duration = end_time - start_time
                    subtask_data.append([task, subtask, start_time, end_time, duration])
    return subtask_data

# Extract subtask information for train and valid datasets
train_subtask_data = extract_subtask_info(data_train_path, subtask_list)
valid_subtask_data = extract_subtask_info(data_valid_path, subtask_list)

# Combine train and valid data
combined_subtask_data = train_subtask_data + valid_subtask_data

# Create a DataFrame for the combined data
df_subtasks = pd.DataFrame(combined_subtask_data, columns=["Task", "Subtask", "Start Time", "End Time", "Duration"])

# Save to CSV
output_csv_path = os.path.join(DATA_PATH, 'subtask_durations.csv')
df_subtasks.to_csv(output_csv_path, index=False)

print(f"Subtask durations saved to: {output_csv_path}")
