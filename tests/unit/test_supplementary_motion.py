from src.perception.supplementary_motion import map_plate_rows
import numpy as np,types
from src.perception.supplementary_reader import SupplementaryReader
from src.perception.supplementary_tracking import read_video_with_motion
from src.utils.types import SignDetection


def test_ocr_result_follows_same_track_when_primary_moves_away_from_old_box():
    key=('P.131a',3)
    row={'primary_label':key[0],'primary_track_id':key[1],'bbox_xyxy':[10,32,30,40],
         'time_windows':[['09:00','16:00']],'human_verified':False}
    result=map_plate_rows([row],{key:[10,10,30,30]},{key:[40,15,60,35]},(100,150,3))
    assert result[0]['bbox_xyxy']==[40,37,60,45]
    assert result[0]['source_bbox_xyxy']==row['bbox_xyxy']
    assert not result[0]['association_verified'] and not result[0]['human_verified']


def test_motion_never_copies_readings_to_another_track_or_implausible_scale():
    key=('P.130',3);row={'primary_label':'P.130','primary_track_id':3,'bbox_xyxy':[10,32,30,40]}
    old={key:[10,10,30,30]}
    assert map_plate_rows([row],old,{('P.130',4):[40,15,60,35]},(100,150,3))==[]
    assert map_plate_rows([row],old,{key:[20,5,120,95]},(100,150,3))==[]


def test_detection_order_and_loss_of_other_sign_do_not_erase_surviving_plate():
    reader=SupplementaryReader();reader.read_video=types.MethodType(read_video_with_motion,reader)
    reader.closed=False;reader.worker=object()  # no worker needed for the completed-evidence fixture
    image=np.zeros((100,160,3),np.uint8)
    a=SignDetection('P.130',.9,(10,10,30,30),track_id=1)
    b=SignDetection('P.131a',.9,(60,10,80,30),track_id=2)
    reader.read_video(image,[a,b],stream_id='s',timestamp=1)
    version=reader.streams['s']['version']
    reader.completed['s']={'version':version,'timestamp':1,'boxes':{('P.131a',2):b.bbox_xyxy},
        'rows':[{'primary_label':'P.131a','primary_track_id':2,'bbox_xyxy':[60,32,80,45],'time_windows':[['09:00','16:00']]}]}
    assert reader.read_video(image,[b,a],stream_id='s',timestamp=1.2)[0]['time_windows']
    assert reader.streams['s']['version']==version
    assert reader.read_video(image,[b],stream_id='s',timestamp=1.4)[0]['primary_track_id']==2
    assert reader.read_video(image,[],stream_id='s',timestamp=1.6)==[]
    assert 's' not in reader.completed
