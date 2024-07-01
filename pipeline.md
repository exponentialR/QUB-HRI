**QUB-PHEO Pipeline**

Our dataset follows the following pipeline:

| No. | **Task Description**                                  | **Script/Command**                            |
|-----|-------------------------------------------------------|-----------------------------------------------|
| 1.  | **Setup Calibration folder**                          | calibration/setup_calib_direc.py              |
| 2.  | **Data-specific calibration**                         | calibration/data_specific_calib.py            |
| 3.  | **Video Tasks Renaming**                              | metadata/video_tasks_rename.py                |
| 4.  | **Reduce resolution**                                 | preprocessing/reduceResolutionParticipants.py |
| 5.  | **Synchronisation**                                   | preprocessing/sync_videos.py                  |
| 6.  | **Add Audio to Aerial Videos**                        | preprocessing/add_audio.py                    |
| 7.  | **Annotate the Aerial Video**                         | actionLabelling/label_studio.sh               |
| 8.  | **Use Timeseries Label to annotate the other videos** |                                               |
| 9.  | **Stereo Calibration of CAM_LL and CAM_LR**           | reconstruction/camera_pair_stereocalib.py     |     
| 10. | **Reconstruction of 3D points**                       | reconstruction/3D_triangulation.py            |
