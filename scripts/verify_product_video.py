"""Verify the completed replay and record evidence without rerunning inference."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import cv2
import imageio_ffmpeg
import numpy as np
import requests

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output/product_20261010'
summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
video=out/'test_video_annotated.mp4'
cap=cv2.VideoCapture(str(video))
fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
assert cap.isOpened() and abs(fps-summary['output_fps'])<.01
assert summary['complete'] and frames==summary['source_frames']
snapshots=[]
for seconds in [9,89,frames/fps-1]:
    cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000);ok,image=cap.read();assert ok
    path=out/f'verified_{round(seconds)}.jpg';cv2.imwrite(str(path),image);snapshots.append(path.name)
cap.release()
assert summary['speech_events'],'No speech was synthesized'
event=summary['speech_events'][0]
pcm=subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-ss',str(event['timestamp']),'-i',str(video),
    '-t','4','-vn','-f','s16le','-ac','1','-ar','24000','-'])
samples=np.frombuffer(pcm,dtype=np.int16)
assert len(samples)>0 and np.max(np.abs(samples.astype(np.int32)))>100
response=requests.get('http://127.0.0.1:8000/pipeline/retrained-video',headers={'Range':'bytes=0-1023'},timeout=30)
assert response.status_code==206 and len(response.content)==1024
current=requests.get('http://127.0.0.1:8000/pipeline/retraining',timeout=30).json()
assert current['product_video']['complete']
tests=ET.parse(out/'tests.xml').getroot()
test_count=sum(int(s.get('tests',0)) for s in tests.findall('testsuite'))
assert all(int(s.get('failures',0))==0 and int(s.get('errors',0))==0 for s in tests.findall('testsuite'))
verification={'output_fps':fps,'output_frames':frames,'duration_seconds':frames/fps,'audio_peak':int(np.max(np.abs(samples.astype(np.int32)))),
    'speech_events':len(summary['speech_events']),'tts_failures':len(summary['tts_failures']),'range_status':response.status_code,'python_tests':test_count,'snapshots':snapshots}
(out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
report=f'''# Kiểm thử video và TTS — 10/10/2026

Đã sửa lỗi mất tên khi crop tốt nhất hết hạn dù model vẫn nhận đúng ở frame mới. Giữ ảnh nguồn 1920 cho phân loại biển nhỏ, không giảm ngưỡng nhận dạng 0,63.

Đã bổ sung ghép track khi biển nhỏ dịch chuyển nhanh giữa hai lần phân tích: chỉ nối khi cùng nhãn được model nhận dạng, vị trí gần, kích thước tương đồng và không có ghép mơ hồ. Trên cùng các box đã lưu, số lượt chưa nhận dạng giảm từ 1.085 xuống {summary['counts']['unknown']:,}; đây là độ phủ tên qua các lần lấy mẫu, không phải độ chính xác hay số biển riêng biệt.

## Kết quả

- Video hoàn chỉnh: {frames:,} khung hình, {fps:g} FPS, {frames/fps:.1f} giây. Mỗi khung hình nguồn được giữ; suy luận {summary['inference_hz']:g} Hz, lớp phủ giữ tới lần phân tích tiếp theo. Đây không phải 30 lượt suy luận/giây.
- {len(summary['speech_events'])} lượt Edge TTS được ghép đúng mốc video; {len(summary['tts_failures'])} lỗi TTS trong lượt xuất. Âm thanh gốc được thay bằng TTS.
- Phát video trực tiếp đã quan sát trên trình duyệt khoảng 29,6–29,8 FPS, 9,3–9,9 lượt nhận diện/giây, phản hồi 83–108 ms khi đồng thời chạy xuất video. Có 7 khung hình phát bị bỏ trong lượt quan sát; không tuyên bố tuyệt đối hết giật trên mọi máy.
- Phân tích offline: trung bình {summary['processing_ms']['mean']:.2f} ms, P95 {summary['processing_ms']['p95']:.2f} ms; không gồm encode và TTS.
- {test_count} test Python và 6 test JavaScript qua; Vite build thành công.
- HTTP Range 206; giải mã được đầu, giữa và cuối video; kiểm tra track âm thanh có tín hiệu.

## Nhận dạng

Trên 28 crop biển được AI rà soát, ảnh JPEG 1280 khớp 26/28, ảnh nguồn 1920 khớp 28/28. Cả hai vẫn nhận nhầm 2/22 crop nền. Không dùng những con số này làm human-verified accuracy. Xem `resolution_comparison.json`.

Các nhãn chưa đủ chắc vẫn không được tự gán tên. Đã giữ yêu cầu hai frame đồng thuận, xóa bằng chứng khi mất detection/tua/xung đột, loại background và ảnh thiếu thông tin.

## Ba logic và giọng đọc

Ghép biển chính/phụ trước → lọc loại xe → khớp hình học làn → tạo câu đọc. Biển dành cho xe khác hoặc biển overhead ở làn bên cạnh bị bỏ qua. Biển phụ chưa đọc được ngăn kết luận áp dụng. Các nhánh này có test hồi quy.

Model hiện có 30 lớp biển chính; chưa tự nhận biết biển phụ xe/giờ và vị trí treo đáng tin cậy. Vì vậy metadata ngữ cảnh vẫn cần nguồn được xác minh. Trên video chưa đủ bằng chứng, TTS chỉ đọc quan sát kèm “chưa xác định đủ điều kiện áp dụng”, không giả định áp dụng cho ô tô. Chưa đọc giá trị số tốc độ/chiều cao hoặc có nhãn người duyệt/holdout mới.

TTS không giữ khóa suy luận, không tạo hàng đợi cảnh báo cũ, chống lặp 20 giây, bỏ kết quả âm thanh quá cũ và hủy khi dừng/tua. Cache 30 tên biển đã tạo thành công. API TTS đo một lần chưa cache 5,45 giây, có cache 7,89 ms. Thời điểm ghép TTS offline không phải độ trễ cảnh báo trực tiếp.

## Tệp

- `test_video_annotated.mp4`: video cuối.
- `summary.json`, `predictions.jsonl`: số đo và dự đoán.
- `verification.json`, `tests.xml`, `live-ui.png`: bằng chứng kiểm thử.
'''
(out/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(verification,indent=2))
