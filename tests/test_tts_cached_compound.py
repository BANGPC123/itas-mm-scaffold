import hashlib
import subprocess
import wave

import imageio_ffmpeg
import numpy as np

from src.interaction.tts_engine import TtsEngine


def test_compound_observation_uses_cached_audio_without_network(tmp_path, monkeypatch):
    import edge_tts

    def unavailable(*args, **kwargs):
        raise AssertionError('Cached observations must not request network speech')

    monkeypatch.setattr(edge_tts, 'Communicate', unavailable)
    engine = TtsEngine({'tts': {'output_dir': str(tmp_path)}})
    parts = [f'Phát hiện biển: {name}. Chưa xác định đủ điều kiện áp dụng.'
             for name in ('Cấm đỗ xe', 'Trẻ em')]
    samples = (np.sin(np.arange(6000) * 2 * np.pi * 440 / 24000) * 8000).astype(np.int16)
    for part in parts:
        key = hashlib.sha256(f'{engine.voice}|{engine.rate}|{part}'.encode()).hexdigest()
        with wave.open(str(tmp_path / f'{key}.mp3'), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(samples.tobytes())
    result = engine.synthesize(' '.join(parts))
    pcm = subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', result,
                                   '-f', 's16le', '-ac', '1', '-ar', '24000', '-'])
    assert len(pcm) == 24000  # Both quarter-second sentences are preserved.
    assert engine.synthesize(' '.join(parts)) == result
