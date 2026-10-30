"""Vietnamese *observation* narration for MP4 capstone research demos.

Never says a sign legally applies or tells the driver how to behave. The
existing reviewed-only sign_speech() remains an independent stronger policy.
"""
from __future__ import annotations
from src.perception.sign_catalog import NAMES_VI

def ai_observation(signs, *, video=False, min_confidence=.74, min_detector=.55, min_video_short_side=32):
    """Returns brief neutral Vietnamese text if a known sign is well supported.

    In video, require temporal multi-frame recognition. Low-quality detections,
    unknown labels, unverified supplementary semantics and ads are abstained.
    This is a *hypothesis*, never ground truth.
    """
    choices=[]
    visited=set()
    for s in signs:
        code=s.label
        if code=='unknown' or code not in NAMES_VI or code in visited:
            continue
        if s.recognition_confidence is None or s.recognition_confidence<min_confidence:
            continue
        if s.confidence<min_detector or s.recognition_source=='background_rejected':
            continue
        if video and s.recognition_source!='temporal_best_crop':
            continue
        if video:
            x1,y1,x2,y2=s.bbox_xyxy
            if min(x2-x1,y2-y1)<min_video_short_side:continue
        if s.panel_role=='supplementary':
            continue
        visited.add(code)
        choices.append(f'{code}: {NAMES_VI[code]}')
        if len(choices)>=2:break
    if not choices:return ''
    return ('AI dự đoán có thể là biển '+
            '; '.join(choices)+
            '. Đây chỉ là nhận diện tham khảo, chưa xác minh điều kiện áp dụng.')


def video_speech_events(signs,t_seconds,announced):
    """Observation cues for tracks not announced yet, at output time t_seconds.

    `announced` holds track ids already cued and is updated in place, so each
    physical track is narrated at most once per offline video.
    """
    events=[]
    for s in signs:
        if s.track_id is None or s.track_id in announced:continue
        text=ai_observation([s],video=True)
        if not text:continue
        announced.add(s.track_id)
        events.append({'t_seconds':round(t_seconds,3),'codes':[s.label],'text':text})
    return events
