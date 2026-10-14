"""Run the deployed pipeline over a local video at a recorded sampling rate."""
from pathlib import Path
import argparse
import base64
import csv
from collections import Counter
from dataclasses import asdict
import json
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.orchestrator import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'output/video_tests/test_video')
    parser.add_argument('--sample-fps', type=float, default=3)
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--road-bottom', type=float, default=1.)
    parser.add_argument('--temporal', action='store_true', help='Use the same temporal confirmation as the live video UI')
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error('--sample-fps must be positive')
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {source}')
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0:
        raise RuntimeError('Video has no usable frame rate')
    step = max(1, round(source_fps/args.sample_fps))
    output_fps = source_fps/step
    scale = min(1, 1280/source_width)
    width, height = (round(source_width*scale)//2*2, round(source_height*scale)//2*2)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    target = out/'test_video_annotated.mp4'
    partial = out/'test_video_annotated.partial.mp4'
    started = time.perf_counter()
    pipeline = Pipeline()
    if pipeline.sign_detector._model is None or pipeline.sign_recognizer.model is None or pipeline.lane_detector.learned is None:
        raise RuntimeError('All three model checkpoints must be available')
    label_counts = Counter()
    records = []
    total_boxes = known_boxes = with_signs = with_known = with_lanes = with_ego = 0
    read_frames = 0
    first_known = first_unknown = False
    max_sign_count = -1
    snapshots = []
    with (out/'encoder.log').open('wb') as encoder_log, (out/'predictions.jsonl').open('w', encoding='utf-8') as predictions:
        encoder = subprocess.Popen([
            ffmpeg, '-y', '-hide_banner', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-r', str(output_fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '23', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log)
        try:
            frame_id = -1
            while capture.grab():
                frame_id += 1
                timestamp = frame_id/source_fps
                if args.max_seconds is not None and timestamp >= args.max_seconds:
                    break
                read_frames += 1
                if frame_id % step:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f'Failed to decode sampled frame {frame_id}')
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                stream_args={'stream_id':'render-video','frame_timestamp':timestamp} if args.temporal else {}
                result = pipeline.run(frame, 10.7769, 106.7009, synthesize_audio=False, road_bottom=args.road_bottom, annotate=False, **stream_args)
                signs = result.perception.signs
                known = [sign for sign in signs if sign.label != 'unknown']
                lane_count = len(result.perception.lane.polylines)
                ego = len(result.perception.lane.ego_boundaries) == 2
                row = {'frame_id': frame_id, 'timestamp_seconds': round(timestamp, 3),
                       'detections': len(signs), 'known_predictions': len(known),
                       'unknown_predictions': len(signs)-len(known), 'lane_boundaries': lane_count,
                       'ego_geometry_available': ego, 'pipeline_ms': result.timings_ms['total']}
                records.append(row)
                predictions.write(json.dumps({**row, 'signs': [asdict(sign) for sign in signs], 'lane': asdict(result.perception.lane)}, ensure_ascii=False)+'\n')
                total_boxes += len(signs)
                known_boxes += len(known)
                with_signs += bool(signs)
                with_known += bool(known)
                with_lanes += lane_count > 0
                with_ego += ego
                label_counts.update(sign.label for sign in signs)
                annotated = frame.copy()
                for points in result.perception.lane.polylines:
                    cv2.polylines(annotated, [np.array(points,np.int32)], False, (70,200,70), 2)
                for sign in signs:
                    x1,y1,x2,y2 = map(int,sign.bbox_xyxy)
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,190,255),2)
                    cv2.putText(annotated,sign.label,(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,190,255),1,cv2.LINE_AA)
                if args.road_bottom < 1:
                    cv2.line(annotated,(0,int(height*args.road_bottom)),(width-1,int(height*args.road_bottom)),(255,255,255),1)
                cv2.rectangle(annotated, (0, 0), (width, 52), (20, 20, 20), -1)
                cv2.putText(annotated, f'{int(timestamp)//60:02}:{timestamp%60:05.2f} | {output_fps:g} sampled fps | lanes {lane_count} | signs {len(signs)} | {row["pipeline_ms"]:.0f} ms', (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated, 'Academic test | car | simulated GPS | sign applicability unverified | audio off', (12, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 210, 210), 1, cv2.LINE_AA)
                encoder.stdin.write(annotated.tobytes())
                if (len(records)-1) % max(1, round(output_fps*120)) == 0:
                    name = f'frame_{frame_id:06}.jpg'
                    cv2.imwrite(str(out/name), annotated)
                    snapshots.append({'file': name, **row})
                for condition, name in [(bool(known) and not first_known, 'first_known.jpg'),
                                        (len(signs)>len(known) and not first_unknown, 'first_unknown.jpg'),
                                        (len(signs)>max_sign_count, 'most_detections.jpg')]:
                    if condition:
                        cv2.imwrite(str(out/name), annotated)
                first_known |= bool(known)
                first_unknown |= len(signs)>len(known)
                max_sign_count = max(max_sign_count, len(signs))
                if len(records) % 100 == 0:
                    progress = {'sampled_frames': len(records), 'video_seconds': round(timestamp, 1), 'elapsed_seconds': round(time.perf_counter()-started, 1)}
                    (out/'progress.json').write_text(json.dumps(progress, indent=2))
                    print(json.dumps(progress), flush=True)
        finally:
            capture.release()
            encoder.stdin.close()
            code = encoder.wait(timeout=60)
        if code:
            raise RuntimeError(f'FFmpeg failed with {code}; see encoder.log')
    if not records:
        raise RuntimeError('No frames processed')
    partial.replace(target)
    with (out/'frames.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    times = np.array([row['pipeline_ms'] for row in records[1:]])
    if not len(times):
        times = np.array([records[0]['pipeline_ms']])
    summary = {'source': str(source), 'source_bytes': source.stat().st_size, 'road_bottom': args.road_bottom, 'temporal_confirmation':args.temporal,
               'source_fps': source_fps, 'source_frames': declared_frames,
               'source_size': [source_width, source_height], 'source_duration_seconds': declared_frames/source_fps,
               'read_source_frames': read_frames, 'sample_every_n_frames': step, 'sampled_fps': output_fps,
               'sampled_frames': len(records), 'output_size': [width, height], 'output_video': str(target),
               'complete_source_scan': args.max_seconds is None and read_frames >= declared_frames,
               'frames_with_detections': with_signs, 'frames_with_known_predictions': with_known,
               'frames_with_lane_boundaries': with_lanes, 'frames_with_ego_geometry': with_ego,
               'detection_observations': total_boxes, 'known_code_observations': known_boxes,
               'unknown_observations': total_boxes-known_boxes, 'label_observations': dict(label_counts.most_common()),
               'pipeline_mean_ms_excluding_first': float(times.mean()), 'pipeline_p95_ms_excluding_first': float(np.percentile(times, 95)),
               'elapsed_wall_seconds': time.perf_counter()-started, 'snapshots': snapshots,
               'limits': ['No ground-truth labels: counts are predictions, not accuracy.',
                          'Repeated observations are not unique physical signs.',
                          'Unseen sign classes can receive an incorrect known code.',
                          'TTS disabled; output video has no audio. GPS is simulated.',
                          'No checkpoint trained on this video; road ROI calibrated using this video.']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'snapshots'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
