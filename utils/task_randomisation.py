import os
import random
import csv

random.seed(42)

# The tasks dictionary
tasks = {
    'BIAH': ['BlockSwap', 'RandomBase', 'Bimanual'],
    'BRIDGE': ['AlternatingSides', 'Bimanual'],
    'CALIBRATION': [],
    'STAIRWAY': ['AlternatingPointer', 'MidAirStack'],
    'TOWER': ['MidAirStack', 'AlternatingHandover']
}

# Create main directory
os.makedirs("PHEO-2.5D", exist_ok=True)

# Prepare for csv file writing
with open('randomisation.csv', 'w', newline='') as csvfile:
    # Number of tasks
    num_tasks = len(tasks.keys())

    # Generate fieldnames
    fieldnames = ['Participant'] + [f'Task_{i + 1}' for i in range(num_tasks)]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # For every participant
    for i in range(8, 101):
        # Create directory for participant
        participant = f'p{str(i).zfill(3)}'  # zfill is used to prepend necessary zeroes
        os.makedirs(f"PHEO-2.5D/{participant}", exist_ok=True)

        # Get task keys and remove 'CALIBRATION', shuffle rest and add 'CALIBRATION' at start
        task_keys = list(tasks.keys())
        task_keys.remove('CALIBRATION')
        random.shuffle(task_keys)
        task_keys.insert(0, 'CALIBRATION')

        # Write participant and their task keys to csv
        row_dict = {'Participant': participant}
        for j, task in enumerate(task_keys):
            row_dict[f'Task_{j + 1}'] = task
        writer.writerow(row_dict)
