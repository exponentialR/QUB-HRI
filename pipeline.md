**QUB-PHEO Pipeline**

## Landmark visualisation

For a dataset copy on another machine, configure `.env` from `.env.example` and start `visualise.py`. See [the visualiser setup guide](docs/visualiser.md); extraction logs and model checkpoints are not required.

Source clips use `videos/<task>/<video-stem>.mp4`; landmarks use `landmarks/<task>/<video-stem>.h5`, following the CAM_AV filename layout. UL/UR contents use the versioned landmark/object schema. The viewer supports synchronized AV/UL/UR/LL/LR playback. Historical LL/LR pixels can be copied into this layout with additive normalized arrays using [the lower-view import command](docs/visualiser.md#normalize-and-import-historical-lower-views). The table below describes the historical preprocessing pipeline.

For existing UL/UR extraction archives, [the landmark file tools](docs/landmark_file_tools.md) provide validation, organization into the video-filename layout, additive normalization and coverage reports. These commands work on saved outputs and require their collection records.

Before a new UL/UR run, [the inventory and source-snapshot utilities](docs/landmark_inventory.md) can record the explicitly selected video scope and retain the package source used by the run. Inventory reports contain metadata and do not replace decoded-frame or synchronization checks.

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
