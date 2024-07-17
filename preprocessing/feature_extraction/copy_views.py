import os
import shutil
from tqdm import tqdm

VIEW_TO_COPY = 'CAM_AV'
super_dir = '/home/samueladebayo/Documents/PhD/QUBPHEO/landmark'
output_root = os.path.join('/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK', VIEW_TO_COPY)
os.makedirs(output_root, exist_ok=True)
count = 0

sub_dirs = [d for d in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, d))]
for sub_dir in tqdm(sub_dirs, desc="Processing directories"):
    proj_dir = os.path.join(super_dir, sub_dir)
    output_dir = os.path.join(output_root, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(proj_dir) if f.endswith('.h5') and f.split('.h5')[0].split('-')[1] == 'CAM_AV']
    for file in tqdm(files, desc=f"Copying files in {sub_dir}"):
        original_file = os.path.join(proj_dir, file)
        output_file = os.path.join(output_dir, file)
        shutil.copyfile(original_file, output_file)
        count += 1

print(f"Total files copied: {count} FOR VIEW: {VIEW_TO_COPY}")
