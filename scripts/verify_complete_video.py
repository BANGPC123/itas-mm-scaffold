"""Decode every exported frame and check evidence/audio before publication."""
from pathlib import Path
import argparse,hashlib,json,math,subprocess,time,wave
import cv2,imageio_ffmpeg,numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    root=args.directory;target=root/'verification.json'
    if target.exists():raise FileExistsError('Verification evidence is immutable')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    video=root/'test_video_annotated.mp4';checks={};started=time.perf_counter()
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    process=subprocess.run([ffmpeg,'-v','error','-i',str(video),'-map','0:v:0',
        '-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True)
    (root/'decode_progress.txt').write_text(process.stdout+process.stderr,encoding='utf-8')
    frames=[int(x.split('=',1)[1]) for x in process.stdout.splitlines() if x.startswith('frame=')]
    decoded=frames[-1] if frames else 0
    checks['full_decode_without_errors']=process.returncode==0 and not process.stderr.strip()
    checks['every_source_frame_exported']=decoded==summary['source_frames']==summary['output_frames']
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    checks['native_video_fps']=abs(fps-summary['output_fps'])<.001
    checks['video_dimensions']=[int(cap.get(3)),int(cap.get(4))]==summary['output_size'];cap.release()
    rows=0;aligned=True;bounds=True;legal_clear=True;timestamps=[]
    with (root/'predictions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);rows+=1;a=row['after'];b=row['before'];timestamps.append(a['timestamp'])
            aligned &= abs(a['timestamp']-b['timestamp'])<1e-6
            legal_clear &= not bool(a['speech'])
            h,w=summary['source_analysis_size'][1],summary['source_analysis_size'][0]
            for polyline in a['lane']['display_polylines'] or []:
                bounds &= all(math.isfinite(x) and math.isfinite(y) and 0<=x<w and 0<=y<h for x,y in polyline)
    expected_samples=math.ceil(summary['source_frames']/round(summary['output_fps']/summary['analysis_hz']))
    checks['all_analysis_timestamps_aligned']=aligned and rows==expected_samples
    checks['lane_coordinates_in_source_bounds']=bool(bounds)
    checks['no_unverified_legal_speech']=bool(legal_clear)
    with wave.open(str(root/'speech.wav'),'rb') as stream:
        rate=stream.getframerate();samples=np.frombuffer(stream.readframes(stream.getnframes()),np.int16)
    audible=[]
    for event in summary['speech_events']:
        chunk=samples[round(event['timestamp']*rate):round(event['end']*rate)]
        audible.append(bool(len(chunk) and np.max(np.abs(chunk.astype(np.int32)))>100))
    checks['all_tts_events_audible']=bool(audible) and all(audible)
    checks['audio_covers_full_video']=abs(len(samples)/rate-summary['duration_seconds'])<1/rate
    # Decode the actual AAC track too, rather than relying on the source WAV.
    audio=subprocess.run([ffmpeg,'-v','error','-i',str(video),'-map','0:a:0',
        '-f','s16le','-ac','1','-ar',str(rate),'-'],capture_output=True)
    encoded=np.frombuffer(audio.stdout,np.int16)
    checks['encoded_audio_decodes']=audio.returncode==0 and not audio.stderr and len(encoded)>0
    checks['encoded_audio_audible']=bool(len(encoded) and np.max(np.abs(encoded.astype(np.int32)))>100)
    checks['inference_and_tts_errors_zero']=not summary['failures'] and not summary['tts_failures']
    digest=hashlib.sha256()
    with video.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    report={'passed':all(checks.values()),'checks':checks,'decoded_frames':decoded,
        'analysis_samples':rows,'source_frames':summary['source_frames'],'video_sha256':digest.hexdigest(),
        'video_size':video.stat().st_size,'video_mtime_ns':video.stat().st_mtime_ns,
        'speech_events_audible':sum(audible),'decode_wall_seconds':time.perf_counter()-started,
        'visual_accuracy_certified':False,'independent_video_gt':False}
    target.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
