from preprocessing.ul_ur_landmarks.make_hand_holdout import choose_rows


def test_challenge_selection_excludes_entire_prior_participants():
    rows=[]
    for index in range(8):
        for subtask in ('A','B','C'):
            rows.append({'pid':f'P{index:02d}','pair_id':f'{subtask}/P{index:02d}-TASK',
                         'subtask_dir':subtask,'views':{view:{'duration_s':4.,'declared_frames':90}
                                                        for view in ('CAM_UL','CAM_UR')}})
    chosen=choose_rows(rows,{'P00','P01'})
    assert len(chosen)==10
    ids={r['pid'] for r in chosen}
    assert len(ids)==5 and not ids&{'P00','P01'}
    assert all(len({r['subtask_dir'] for r in chosen if r['pid']==pid})==2 for pid in ids)
    assert chosen==choose_rows(list(reversed(rows)),{'P00','P01'})
