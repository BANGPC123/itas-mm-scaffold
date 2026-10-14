import sys
from types import SimpleNamespace
import pytest
from src.interaction.tts_engine import TtsEngine

def test_edge_tts_is_cached_by_voice_rate_and_text(tmp_path,monkeypatch):
    calls=[]
    class Communicate:
        def __init__(self,*a,**kw):calls.append((a,kw))
        async def save(self,path):
            from pathlib import Path
            Path(path).write_bytes(b'mp3-test')
    monkeypatch.setitem(sys.modules,'edge_tts',SimpleNamespace(Communicate=Communicate))
    engine=TtsEngine({'tts':{'output_dir':str(tmp_path)}})
    first=engine.synthesize('xin chao');second=engine.synthesize('xin chao')
    assert first==second and len(calls)==1

def test_failed_tts_does_not_leave_cache_entry(tmp_path,monkeypatch):
    class Communicate:
        def __init__(self,*a,**kw):pass
        async def save(self,path):raise RuntimeError('network')
    monkeypatch.setitem(sys.modules,'edge_tts',SimpleNamespace(Communicate=Communicate))
    engine=TtsEngine({'tts':{'output_dir':str(tmp_path)}})
    with pytest.raises(RuntimeError):engine.synthesize('xin chao')
    assert list(tmp_path.iterdir())==[]

def test_transient_failure_retries_without_caching_partial_audio(tmp_path,monkeypatch):
    calls=[]
    class Communicate:
        def __init__(self,*a,**kw):pass
        async def save(self,path):
            from pathlib import Path
            calls.append(path)
            if len(calls)==1:
                Path(path).write_bytes(b'partial')
                raise RuntimeError('temporary service failure')
            assert not Path(path).exists()
            Path(path).write_bytes(b'complete-audio')
    monkeypatch.setitem(sys.modules,'edge_tts',SimpleNamespace(Communicate=Communicate))
    engine=TtsEngine({'tts':{'output_dir':str(tmp_path)}})
    from pathlib import Path
    result=Path(engine.synthesize('xin chao'))
    assert len(calls)==2 and result.read_bytes()==b'complete-audio'
    assert list(tmp_path.iterdir())==[result]
