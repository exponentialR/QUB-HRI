import pytest

from preprocessing.ul_ur_landmarks.collection_status import ledger_rows, progress


def test_live_ledger_ignores_only_incomplete_final_line(tmp_path):
    path=tmp_path/'ledger.jsonl'
    path.write_bytes(b'{"status":"written"}\n{"status":')
    assert ledger_rows(path)==([{'status':'written'}],True)
    path.write_bytes(b'broken\n')
    with pytest.raises(ValueError):
        ledger_rows(path)


def test_progress_deduplicates_reruns_and_retains_failed_clips():
    inventory=[{'pair_id':'task/P01-clip','views':{v:{'declared_frames':10} for v in ('CAM_UL','CAM_UR')}}]
    row={'pair_id':'task/P01-clip','view':'CAM_UL','status':'written','frames':10,'elapsed_s':2}
    rows=[row,{**row,'status':'skipped_valid'}, {**row,'view':'CAM_UR','status':'failed','error':'decode'}]
    report=progress(inventory,rows)
    assert report['logged_valid_clips']==1 and report['logged_frames']==10
    assert report['remaining_clips']==1 and not report['all_expected_outputs_logged_valid']
    assert report['logged_processing_s']==2
    assert report['latest_failures'][0]['view']=='CAM_UR'
    with pytest.raises(ValueError):
        progress(inventory,[{**row,'pair_id':'wrong'}])
