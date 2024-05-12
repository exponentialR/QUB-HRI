import os
from reconstruction.downgrade_fps import downgrade_fps, match_frame_length


def adjust_video_fps_frame_count(base_video_path, target_video_path, adjusted_video_path):
    # Placeholder for the actual adjustment logic
    downgraded_video = downgrade_fps(base_video_path, target_video_path)
    if downgraded_video:
        _, _ = match_frame_length(base_video_path, target_video_path)

    pass


class FPSFrameCountRebaser:
    def __init__(self, proj_dir, output_dir, start_participant, end_participant):
        self.proj_dir = proj_dir
        self.output_dir = output_dir
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.base_view = 'CAM_LR'
        self.other_views = ['CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']

    def rebase_videos(self):
        for i in range(self.start_participant, self.end_participant + 1):
            base_participant_dir = os.path.join(self.proj_dir, f'p{i:02d}', self.base_view)
            for vid_file in os.listdir(base_participant_dir):
                if vid_file.endswith('.mp4'):
                    base_video_path = os.path.join(base_participant_dir, vid_file)
                    for cam_view in self.other_views:
                        target_view_dir = os.path.join(self.proj_dir, f'p{i:02d}', cam_view)
                        target_video_path = os.path.join(target_view_dir, vid_file)
                        adjusted_video_path = os.path.join(self.output_dir, f'p{i:02d}', cam_view, vid_file)
                        os.makedirs(os.path.dirname(adjusted_video_path), exist_ok=True)
                        # Call the adjustment function for each video
                        if os.path.exists(target_video_path):
                            adjust_video_fps_frame_count(base_video_path, target_video_path, adjusted_video_path)
