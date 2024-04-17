import os
import os

# Given full path

# Split the path into components



def extract_name(full_path):
    components = full_path.split('/')
    desired_path = ''
    for component in components:
        if component.startswith('p') or component.startswith('CAM_') or component.endswith('.mp4'):
            desired_path = os.path.join(desired_path, component)
    return desired_path

full_path = "/home/qub-hri/Documents/Datasets/QUB-PHEO/p61/CAM_AV/BIAH_BV.mp4"

test = extract_name(full_path)
print(test)