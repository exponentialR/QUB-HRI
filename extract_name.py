import os
file_list = []

for files in os.listdir('/home/qub-hri/Documents/PHEO Waiting Data/UL_camera'):
    if files.endswith('.MP4'):
        file_list.append(files)

print(file_list)
