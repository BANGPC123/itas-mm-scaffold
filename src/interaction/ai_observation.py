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
    codes=[]
    for s in signs:
        code=s.label
        if code=='unknown' or code not in NAMES_VI or code in codes:
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
        codes.append(code)
        if len(codes)>=2:break
    if not codes:return ''
    return observation_text(codes)


def observation_text(codes):
    choices=[f'{code}: {NAMES_VI[code]}' for code in codes]
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


def group_speech_events(events,gap_s=1.5,per_cue=2):
    """Merge cues that follow each other within gap_s into spoken cues of up to per_cue signs.

    A burst of signs is read as one cue at its first time. Signs beyond per_cue are
    read in the next cue, placed at the end of the burst, so no sign is left unspoken.
    """
    bursts=[]
    for event in sorted(events,key=lambda e:e['t_seconds']):
        if bursts and event['t_seconds']-bursts[-1][-1]['t_seconds']<=gap_s:
            bursts[-1].append(event)
        else:
            bursts.append([event])
    grouped=[]
    for burst in bursts:
        codes=list(dict.fromkeys(code for event in burst for code in event['codes']))
        for index,start in enumerate(range(0,len(codes),per_cue)):
            chunk=codes[start:start+per_cue]
            when=burst[0]['t_seconds'] if index==0 else burst[-1]['t_seconds']
            grouped.append({'t_seconds':when,'codes':chunk,'text':observation_text(chunk)})
    return grouped
