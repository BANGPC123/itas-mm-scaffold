"""Automatic vehicle interpretation with explicit unresolved visual conditions.

Machine observations leave the independently reviewed legal flags unchanged.
"""
from src.context.time_scope import time_scope
from src.context.sign_scope_principles import base_vehicle_scope
from src.perception.supplementary_panel_proposals import propose_plates,bbox_iou
from src.perception.sign_catalog import NAMES_VI

NUMERIC={'P.117','P.127'}


def assess_car(sign,image,candidates=(),*,video=False,observed_at=None,camera_view='forward'):
    result={'vehicle':'car','status':'uncertain','vehicle_scope':'unknown',
            'basis':'machine_visual_evidence','human_verified':False,
            'legal_effect_established':False,'conditions':[],'unresolved':[],
            'text':'Ô tô: chưa đủ bằng chứng nhận dạng biển.'}
    if sign.label not in NAMES_VI or sign.panel_role=='supplementary':return result
    if camera_view!='forward':
        result['unresolved']=['camera_direction']
        result['text']=('Ô tô: camera đang quay ngang/vào cabin; chưa xác định biển cho hướng đi hiện tại.'
            if camera_view=='side_or_cabin' else 'Ô tô: chưa đủ bằng chứng xác định hướng nhìn của camera và phạm vi biển.')
        return result
    if video and sign.recognition_source!='temporal_best_crop':return result
    if sign.recognition_confidence is None or sign.recognition_confidence<.74:return result
    x1,y1,x2,y2=sign.bbox_xyxy
    if min(x2-x1,y2-y1)<32:
        result['unresolved']=['primary_detail']
        result['text']='Ô tô: biển quá nhỏ để đọc chắc loại biển và điều kiện kèm theo.'
        return result
    if sign.applicability=='ignored' and sign.context_verified and sign.label_verified:
        result.update(status='excluded',vehicle_scope='excluded',basis='reviewed_context',
                      text='Ô tô: biển không áp dụng theo điều kiện đã xác minh.')
        return result
    result['vehicle_scope']='included' if base_vehicle_scope(sign.label,'car')!='excluded' else 'excluded'
    if sign.label.startswith(('W.','I.','DP.','R.420','R.421')):
        result.update(status='information',text='Ô tô: biển cảnh báo/chỉ dẫn; chưa suy ra lệnh cấm cho xe.')
        return result
    proposals=propose_plates(image,sign.bbox_xyxy,max_candidates=3)
    attached=[row for row in candidates if row.get('primary_label')==sign.label
              and row.get('primary_track_id')==sign.track_id
              and any(bbox_iou(row.get('bbox_xyxy',[0,0,0,0]),p['bbox_xyxy'])>=.5 for p in proposals)]
    paired=[r for r in candidates if r.get('primary_label')==sign.label and r.get('primary_track_id')==sign.track_id]
    unresolved=[];conditions=[];excluded=False
    if paired and not proposals:unresolved.append('supplementary_not_visible')
    for proposal in proposals:
        matches=[r for r in attached if bbox_iou(r['bbox_xyxy'],proposal['bbox_xyxy'])>=.5]
        if not matches:
            unresolved.append('unreadable_supplementary');continue
        row=max(matches,key=lambda r:bbox_iou(r['bbox_xyxy'],proposal['bbox_xyxy']))
        decoded=False
        if row.get('vehicle_variant_agreement') and row.get('vehicle_candidates'):
            vehicles=row['vehicle_candidates'];polarity=row.get('condition_polarity')
            includes_car='car' in vehicles or 'automobile' in vehicles
            if polarity=='exclusion':excluded|=includes_car
            elif polarity in ('inclusion','unspecified'):excluded|=not includes_car
            else:unresolved.append('vehicle_polarity');continue
            result['vehicle_scope']='excluded' if excluded else 'included'
            conditions.append({'type':'vehicles','values':vehicles,'polarity':polarity});decoded=True
        if row.get('variant_agreement') and row.get('time_windows'):
            conditions.append({'type':'time','windows':row['time_windows']});decoded=True
        if not decoded:unresolved.append('unreadable_supplementary')
    if sign.is_compound and any(p=='unknown' or p.startswith('S.') for p in sign.panel_labels):
        unresolved.append('compound_scope')
    if sign.label in NUMERIC:unresolved.append('numeric_limit')
    windows=[c['windows'] for c in conditions if c['type']=='time']
    temporal=time_scope(sign.label,windows,observed_at)
    if temporal is None:unresolved.append('recording_date_time')
    result.update(conditions=conditions,unresolved=sorted(set(unresolved)))
    if excluded:
        result.update(status='excluded_candidate',text='AI: biển phụ loại trừ ô tô con khỏi phạm vi biển chính.')
    elif temporal is False:
        result.update(status='excluded_candidate',text='AI: ô tô đang ngoài ngày/giờ đọc được trên biển phụ.')
    elif unresolved:
        result.update(status='conditional',text='Theo loại xe, biển chính có thể áp dụng cho ô tô; '+
                      ('giới hạn số chưa đọc được.' if 'numeric_limit' in unresolved else
                       'cần xét ngày/giờ ghi hình.' if 'recording_date_time' in unresolved and len(unresolved)==1 else
                       'điều kiện biển phụ/phạm vi hướng đi chưa đọc đủ.'))
    else:
        result.update(status='applies_candidate',text='AI: biển có thể áp dụng cho ô tô; chưa thấy biển phụ hạn chế loại xe.')
    if windows:
        values=', '.join(a+'–'+b for group in windows for a,b in group)
        result['text']+=' Giờ đọc được: '+values+'.'
    return result


def car_summary(signs):
    rows=[]
    for sign in signs:
        context=getattr(sign,'car_context',{})
        if context and sign.label!='unknown':
            text=f'{sign.label}: {context["text"]}'
            if text not in rows:rows.append(text)
    return ' '.join(rows[:3])
