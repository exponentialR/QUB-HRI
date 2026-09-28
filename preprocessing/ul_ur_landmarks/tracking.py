"""Conservative hand identity assignment for the two-actor pilot."""

from __future__ import annotations

import math
import numpy as np

from .schema import Hand


def assign_actor(hand_xy: np.ndarray, participant_hands: list[np.ndarray],
                 participant_wrists: list[np.ndarray], width: int, height: int) -> str:
    """Use participant evidence; preserve 'unknown' for overlap or missing pose."""
    wrist = np.asarray(hand_xy[0], dtype=float)
    if not np.isfinite(wrist).all():
        return "unknown"
    scale = math.hypot(width, height)
    known = [np.asarray(x, dtype=float) for x in participant_hands + participant_wrists]
    known = [x for x in known if x.shape == (2,) and np.isfinite(x).all()]
    if not known:
        return "unknown"
    distance = min(np.linalg.norm(wrist - x) for x in known)
    if distance <= 0.06 * scale:
        return "participant"
    if distance >= 0.16 * scale:
        return "other_actor"
    return "unknown"


class HandTracker:
    def __init__(self, width: int, height: int) -> None:
        self.max_step = 0.08 * math.hypot(width, height)
        self.previous: list[Hand] = []
        self.next_id = 0

    def update(self, hands: list[Hand]) -> list[Hand]:
        unmatched = set(range(len(self.previous)))
        for hand in hands:
            wrist = np.asarray(hand.xy[0], dtype=float)
            matches = []
            for index in unmatched:
                prior = self.previous[index]
                prior_wrist = np.asarray(prior.xy[0], dtype=float)
                if not np.isfinite(wrist).all() or not np.isfinite(prior_wrist).all():
                    continue
                if hand.actor != "unknown" and prior.actor != "unknown" and hand.actor != prior.actor:
                    continue
                distance = float(np.linalg.norm(wrist - prior_wrist))
                if distance <= self.max_step:
                    matches.append((distance, index))
            if matches:
                _, index = min(matches)
                hand.track_id = self.previous[index].track_id
                if hand.actor == "unknown":
                    hand.actor = self.previous[index].actor
                unmatched.remove(index)
            else:
                hand.track_id = self.next_id
                self.next_id += 1
        self.previous = hands
        return hands


class BoxHandTracker:
    """Collection tracker: preserve box observations even when joints are missing.

    Tracks survive two missing frames. Known actor and anatomical side conflicts
    cannot match. Globally sorted center distances avoid detection-order bias.
    This is a geometric tracker; difficult crossings still require measurement.
    """

    def __init__(self, width: int, height: int):
        self.max_step = .08 * math.hypot(width,height)
        self.frame = -1
        self.next_id = 0
        self.tracks = {}

    def update(self,hands:list[Hand]) -> list[Hand]:
        self.frame += 1
        self.tracks = {i:entry for i,entry in self.tracks.items() if self.frame-entry['frame']<=3}
        centers = []
        for hand in hands:
            if hand.bbox_xyxy is not None:
                box = np.asarray(hand.bbox_xyxy,dtype=float)
                center = (box[:2]+box[2:])/2
            else:
                points = hand.xy[np.isfinite(hand.xy).all(axis=1)]
                center = np.median(points,axis=0) if len(points) else np.array([np.nan,np.nan])
            centers.append(center)
        candidates = []
        for j,hand in enumerate(hands):
            if not np.isfinite(centers[j]).all():
                continue
            for track_id,previous in self.tracks.items():
                if hand.actor!='unknown' and previous['actor']!='unknown' and hand.actor!=previous['actor']:
                    continue
                if hand.handedness!='unknown' and previous['handedness']!='unknown' and hand.handedness!=previous['handedness']:
                    continue
                distance = float(np.linalg.norm(centers[j]-previous['center']))
                if distance <= self.max_step * (self.frame-previous['frame']):
                    candidates.append((distance,track_id,j))
        used_tracks,used_hands=set(),set()
        for _,track_id,j in sorted(candidates):
            if track_id in used_tracks or j in used_hands:
                continue
            used_tracks.add(track_id)
            used_hands.add(j)
            hands[j].track_id=track_id
            if hands[j].actor=='unknown':
                hands[j].actor=self.tracks[track_id]['actor']
        for j,hand in enumerate(hands):
            if j not in used_hands:
                hand.track_id=self.next_id
                self.next_id += 1
            if np.isfinite(centers[j]).all():
                self.tracks[hand.track_id]={'center':centers[j],'actor':hand.actor,
                                            'handedness':hand.handedness,'frame':self.frame}
        return hands
