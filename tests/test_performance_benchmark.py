import numpy as np

from preprocessing.ul_ur_landmarks.performance_benchmark import compare
from preprocessing.ul_ur_landmarks.schema import Frame, Hand


def test_output_comparison_detects_missing_points_and_actor_changes():
    first=Frame(0,pose_xy=np.array([[0.,0.],[1.,2.]]),hands=[Hand(np.zeros((21,2)),np.ones(21))])
    second=Frame(0,pose_xy=np.array([[0.,0.],[np.nan,np.nan]]),hands=[Hand(np.zeros((21,2)),np.ones(21),actor='participant')])
    result=compare([first],[second])
    assert result['finite_mask_differences']==2
    assert result['structural_or_label_differences']==1
    assert result['max_abs_delta_by_field']['frame.pose_xy']==0


def test_output_comparison_handles_absent_arrays_and_different_observation_counts():
    first=Frame(0)
    second=Frame(0,pose_xy=np.zeros((133,2)),hands=[Hand(np.zeros((21,2)),np.ones(21))])
    assert compare([first],[second])['structural_or_label_differences']==2
    assert compare([second],[first])['structural_or_label_differences']==2
