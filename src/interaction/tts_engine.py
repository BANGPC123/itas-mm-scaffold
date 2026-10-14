import asyncio
import hashlib
from pathlib import Path
import threading
import uuid
import re
import subprocess

class TtsEngine:
    def __init__(self, interaction_config):
        cfg=interaction_config['tts']
        self.provider=cfg.get('provider',cfg.get('engine','edge-tts'))
        if self.provider not in ('edge-tts','vieneu-onnx'):raise ValueError('Unsupported speech provider')
        self.voice=cfg.get('voice_id','vi-VN-HoaiMyNeural')
        self.rate=cfg.get('rate','+0%')
        self.output_dir=Path(cfg['output_dir'])
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        self.timeout_seconds=float(cfg.get('timeout_seconds',9.5))
        self.offline=None
        self.offline_python=cfg.get('python_path')

    def cache_path(self,text):
        key=hashlib.sha256(f'{self.voice}|{self.rate}|{text}'.encode()).hexdigest()
        return self.output_dir/f'{key}.mp3'

    def warmup(self):
        if self.provider=='vieneu-onnx':
            if self.offline is None:
                from src.interaction.offline_speech import OfflineSpeech
                self.offline=OfflineSpeech(self.output_dir,python_path=self.offline_python,
                    timeout_seconds=self.timeout_seconds)
            self.offline.warmup()

    def close(self):
        if self.offline is not None:self.offline.close()

    def synthesize(self,text):
        target=self.cache_path(text);key=target.stem
        with self.lock:
            if target.is_file() and target.stat().st_size>0:return str(target)
            temp=self.output_dir/f'{key}.{uuid.uuid4().hex}.mp3'
            if self.provider=='vieneu-onnx':
                import imageio_ffmpeg
                wav=temp.with_suffix('.wav')
                try:
                    self.warmup();self.offline.synthesize_wav(text,wav)
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(wav),
                        '-codec:a','libmp3lame','-b:a','96k',str(temp)],check=True,timeout=15,
                        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    if not temp.is_file() or temp.stat().st_size==0:raise RuntimeError('Offline speech returned empty audio')
                    temp.replace(target)
                finally:
                    wav.unlink(missing_ok=True);temp.unlink(missing_ok=True)
                return str(target)
            import edge_tts
            # Multi-sign observations reuse the exact cached sentences, including qualifiers.
            parts=re.split(r'(?<=\.) (?=Phát hiện biển: )',text)
            cached=[self.output_dir/f"{hashlib.sha256(f'{self.voice}|{self.rate}|{part}'.encode()).hexdigest()}.mp3" for part in parts]
            if len(parts)>1 and all(path.is_file() and path.stat().st_size>0 for path in cached):
                import imageio_ffmpeg
                inputs=[arg for path in cached for arg in ('-i',str(path))]
                streams=''.join(f'[{i}:a]' for i in range(len(cached)))
                try:
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error',*inputs,
                        '-filter_complex',f'{streams}concat=n={len(cached)}:v=0:a=1[a]',
                        '-map','[a]','-codec:a','libmp3lame',str(temp)],check=True,timeout=10,
                        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    if not temp.is_file() or temp.stat().st_size==0:raise RuntimeError('Empty combined speech')
                    temp.replace(target)
                    return str(target)
                except (OSError,subprocess.SubprocessError,RuntimeError):
                    temp.unlink(missing_ok=True)
            async def save():
                for attempt in range(2):
                    try:
                        await asyncio.wait_for(edge_tts.Communicate(text,self.voice,rate=self.rate).save(str(temp)),timeout=self.timeout_seconds)
                        if not temp.is_file() or temp.stat().st_size==0:raise RuntimeError('TTS returned empty audio')
                        return
                    except Exception:
                        temp.unlink(missing_ok=True)
                        if attempt:raise
                        await asyncio.sleep(.25)
            try:
                asyncio.run(save())
                if not temp.is_file() or temp.stat().st_size==0:raise RuntimeError('TTS returned empty audio')
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        return str(target)
