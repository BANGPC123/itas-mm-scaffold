import queue
import sys
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.interaction.offline_speech import OfflineSpeech
from src.interaction.tts_engine import TtsEngine


def engine(tmp_path):
    return TtsEngine({'tts': {'engine': 'vieneu-onnx', 'voice_id': 'vieneu-HaiDang',
                              'output_dir': str(tmp_path)}})


def test_offline_speech_is_atomic_and_cached_without_online_provider(tmp_path, monkeypatch):
    speech = engine(tmp_path)
    calls = []
    def save_wav(text, target):
        calls.append(text)
        target.write_bytes(b'wav-test')
    speech.offline = SimpleNamespace(warmup=lambda: None, synthesize_wav=save_wav)
    monkeypatch.setitem(sys.modules, 'edge_tts', None)
    def encode(command, **kwargs):
        Path(command[-1]).write_bytes(b'encoded-audio')
    monkeypatch.setattr('src.interaction.tts_engine.subprocess.run', encode)
    first = speech.synthesize('Tiếng Việt tại máy')
    assert speech.synthesize('Tiếng Việt tại máy') == first
    assert calls == ['Tiếng Việt tại máy']
    assert list(tmp_path.iterdir()) == [Path(first)]


def test_failed_offline_speech_does_not_publish_partial_audio(tmp_path):
    speech = engine(tmp_path)
    def fail(text, target):
        target.write_bytes(b'partial')
        raise TimeoutError('inference deadline')
    speech.offline = SimpleNamespace(warmup=lambda: None, synthesize_wav=fail)
    with pytest.raises(TimeoutError):
        speech.synthesize('Xin chào')
    assert list(tmp_path.iterdir()) == []


def test_timed_out_worker_is_closed_before_another_request(tmp_path, monkeypatch):
    worker = OfflineSpeech(tmp_path)
    worker.messages = queue.Queue()
    closed = []
    monkeypatch.setattr(worker, 'close', lambda: closed.append(True))
    with pytest.raises(TimeoutError):
        worker._receive(.001)
    assert closed == [True]


def test_offline_voice_has_a_different_cache_identity(tmp_path):
    online = TtsEngine({'tts': {'output_dir': str(tmp_path)}})
    assert engine(tmp_path).cache_path('Xin chào') != online.cache_path('Xin chào')


def test_worker_receives_vietnamese_utf8_even_under_a_windows_codepage(tmp_path):
    worker = Path(__file__).resolve().parents[2] / 'scripts/offline_tts_worker.py'
    target = tmp_path / 'test.wav'
    text = 'Đỗ xe, đọc tiếng Việt'
    # Exercise the actual JSON worker without loading model weights. The old
    # code misread UTF-8 byte 0x90 as a surrogate under the Windows codepage.
    bootstrap = '''import sys,io,runpy
from pathlib import Path
from types import SimpleNamespace
class Model:
    def __init__(self,**kwargs):pass
    def infer(self,text,**kwargs):return text
    def save(self,text,path):Path(path).write_text(text,encoding='utf-8')
sys.modules['vieneu']=SimpleNamespace(Vieneu=Model)
sys.stdin=io.TextIOWrapper(sys.stdin.buffer,encoding='cp1252',errors='surrogateescape')
sys.argv=[sys.argv[1],'--output-dir',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    request = json.dumps({'text': text, 'wav_path': str(target)}, ensure_ascii=False).encode('utf-8')
    result = subprocess.run([sys.executable, '-c', bootstrap, str(worker), str(tmp_path)],
                            input=request + b'\n', capture_output=True, check=True)
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages[0]['ready'] and 'error' not in messages[-1]
    assert target.read_text(encoding='utf-8') == text
