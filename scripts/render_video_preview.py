"""Replay sampled predictions over source-rate frames; no frame duplication."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('predictions', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, default=60)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines()]
    capture = cv2.VideoCapture(args.source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise RuntimeError('Cannot open source video')
    encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
        '-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','veryfast','-crf','23','-threads','2',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)], stdin=subprocess.PIPE)
    index = 0
    count = 0
    try:
        while count/fps < args.seconds:
            ok, frame = capture.read()
            if not ok: break
            timestamp = count/fps
            frame = cv2.resize(frame,(1280,720))
            while index+1<len(records) and records[index+1]['timestamp_seconds']<=timestamp:
                index+=1
            record = records[index]
            if 0<=timestamp-record['timestamp_seconds']<=.3:
                for points in record['lane']['polylines']:
                    cv2.polylines(frame,[np.array(points,np.int32)],False,(70,200,70),2)
                for sign in record['signs']:
                    x1,y1,x2,y2=map(int,sign['bbox_xyxy'])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,190,255),2)
                    cv2.putText(frame,sign['label'],(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,190,255),1)
            cv2.rectangle(frame,(0,0),(1280,35),(20,20,20),-1)
            cv2.putText(frame,f'{timestamp:05.1f}s | source playback {fps:g} FPS | saved predictions 3 FPS | ROI 65% | audio off',
                (12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            encoder.stdin.write(frame.tobytes())
            count+=1
    finally:
        capture.release()
        encoder.stdin.close()
        code=encoder.wait(timeout=60)
    if code: raise RuntimeError(f'Encoder exit code {code}')
    verify=cv2.VideoCapture(str(args.output))
    decoded=0
    while verify.grab(): decoded+=1
    actual_fps=verify.get(cv2.CAP_PROP_FPS)
    verify.release()
    assert decoded==count and abs(actual_fps-fps)<.01
    metadata={'frames':count,'playback_fps':actual_fps,'duration_seconds':count/fps,'prediction_fps':3,'audio':False}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata))


if __name__=='__main__': main()
