# Camera Calibration and Video Correction using Charuco Boards

## Overview

This repository provides a set of Python scripts to perform camera calibration using Charuco boards and correct distortions in videos. The calibration parameters are calculated using frames extracted from calibration videos and then applied to other videos for distortion correction.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Calibrating Your Camera](#calibrating-your-camera)
  - [Correcting Videos](#correcting-videos)
- [Scripts](#scripts)
- [Contribute](#contribute)
- [License](#license)

## Installation

Clone this repository to your local machine to get started.

```bash
git clone https://github.com/exponentialR/QUB-HRI/Calibration
```

## Getting Started

### Setting Up the Environment

1. Navigate to the root directory of the repository.
2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Calibrating Your Camera

1. Place your Charuco board calibration videos in the `calibration_videos/` directory.
   
2. Run the `calibrate_camera_from_video.py` script. This script will extract frames from the calibration videos, find the Charuco board corners, and calculate the camera calibration parameters.

    ```bash
    python calibrate_camera_from_video.py
    ```
   
3. Upon successful calibration, a file named `calibration_parameters.npz` will be generated. This file contains the camera matrix and distortion coefficients that will be used for video correction.

### Correcting Videos

1. Place the videos you wish to correct in the `videos_to_correct/` directory.

2. Run the `correct_video.py` script. This script will use the calibration parameters to correct the distortions in each video and save the corrected videos in the `corrected_videos/` directory.

    ```bash
    python correct_video.py
    ```
   
3. Once the script completes, you can find the corrected videos in the `corrected_videos/` directory.

## Scripts

- `calibrate_camera_from_video.py`: This script extracts frames from the Charuco board calibration videos in `calibration_videos/` and calculates the camera calibration parameters.
  
- `correct_video.py`: This script applies the calculated calibration parameters to the videos in `videos_to_correct/` to correct any distortions.

- `setup_directories.py` (Optional): This script sets up the initial directories (`calibration_videos/`, `videos_to_correct/`, `corrected_videos/`) needed for the project.

## Contribute

Feel free to open an issue or make a pull request if you find any bugs or have any suggestions to improve the code.

## License

This project is licensed under the MIT License.
