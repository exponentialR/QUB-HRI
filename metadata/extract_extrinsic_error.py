import os
import numpy as np


def load_calib_error(calibration_path):
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file not found at {calibration_path}")
    try:
        with np.load(calibration_path) as data:
            error = data['retval']
            return error
    except IOError:
        print(f'Error Loading Claibration data from {calibration_path}')
        return None


def participant_calib_error(proj_dir, start_part, end_part, output_csv):
    """
    Extract calibration error for a range of participants and saves into csv file for inspection
    Args:
        proj_dir:
        start_part:
        end_part:

    Returns:

    """
    calibration_list = [os.path.join(proj_dir, f'p{p:02d}', f'p{p:02d}_LL_LR_stereo.npz') for p in range(start_part, end_part + 1)]
    print(calibration_list)
    for calib_path in calibration_list:
        if not os.path.exists(calib_path):
            print(f"Calibration file not found at {calib_path}")
            # save into csv file and skip
            with open(output_csv, 'a') as f:
                f.write(f"{calib_path},N/A\n")
            continue
        else:
            # Log the calibration error into a csv file
            error = load_calib_error(calib_path)
            print(f"Calibration error for {calib_path}: {error}")
            with open(output_csv, 'a') as f:
                f.write(f"{calib_path},{error}\n")


if __name__ == '__main__':
    proj_dir = '/home/samueladebayo/Documents/PhD/QUBPHEO/calibration'
    start_part = 1
    end_part = 70
    output_csv = '/home/samueladebayo/Documents/PhD/QUBPHEO/calibration_error.csv'
    participant_calib_error(proj_dir, start_part, end_part, output_csv)
