import os
from pathlib import Path
from tqdm import tqdm

consented_participants = (
    "P01","P02","P03","P04","P05","P06","P07","P08","P09",
    "P11","P12","P14","P15","P17","P18","P19",
    "P21","P22","P23","P24","P25","P26","P29","P30","P31","P32","P33",
    "P35","P36","P37","P40","P44","P45","P46","P47","P48","P50",
    "P52","P53","P54","P55","P57","P58","P60",
    "P64","P65","P67","P68","P69","P70"
)
EXT = ('mp4', 'MP4', 'avi', 'AVI', 'mov', 'MOV')

if __name__ == '__main__':
    c_participants = sorted([i.lower() for i in consented_participants])
    DATA_PATH = Path(input("Please enter the absolute path of QUB-PHEO dataset, e.g. /home/alien_arise/Documents/segmented"))
    consented_participants = [i.lower() for i in consented_participants]
    subtask_list = sorted([p for p in DATA_PATH.iterdir() if p.is_dir()])

    subtask_dict = {}
    for subtask_dir in tqdm(subtask_list, desc='going through subtask'):

        subtask_vid_files = sorted([str(f) for f in subtask_dir.iterdir() if f.is_file()
                                    and str(f).lower().endswith(EXT) and any(pid in f.stem.lower() for pid in c_participants)])
        subtask_dict[str(os.path.basename(subtask_dir))] = len(subtask_vid_files)

    print(f'Subtask Consented Videos statistics: \n'
          f'{subtask_dict}')