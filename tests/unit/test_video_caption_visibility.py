from pathlib import Path
import cv2,numpy as np
from PIL import ImageFont
from src.perception.video_text_overlay_safe import SafeVideoTextOverlay


def test_captions_keep_top_edge_sign_and_supplementary_text_visible():
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf' if Path('C:/Windows/Fonts/arial.ttf').is_file() else 'DejaVuSans.ttf',18)
    frame=np.full((720,1280,3),90,np.uint8)
    cv2.circle(frame,(1220,35),25,(200,40,20),-1)
    cv2.rectangle(frame,(1180,75),(1260,120),(240,240,240),-1)
    cv2.putText(frame,'09-16',(1188,98),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
    payload={'lane':{'camera_view':'forward'},'signs':[{'label':'P.131a','car_context':{}}]}
    result=SafeVideoTextOverlay(font,30,3).draw(frame.copy(),payload,[],[(1180,5,1260,65)],149.333)
    assert np.array_equal(result[10:60,1195:1245],frame[10:60,1195:1245])
    assert np.array_equal(result[78:116,1185:1255],frame[78:116,1185:1255])
