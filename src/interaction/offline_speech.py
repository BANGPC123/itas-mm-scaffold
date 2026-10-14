"""Own one isolated CPU ONNX speech worker without changing the CUDA runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading


class OfflineSpeech:
    def __init__(self, output_dir, *, python_path=None, startup_seconds=180, timeout_seconds=30):
        self.root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir).resolve()
        default = self.root / '.tts-venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        self.python = Path(python_path) if python_path else default
        self.startup_seconds = float(startup_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.process = None
        self.log = None
        self.messages = None

    @staticmethod
    def _read_output(process, messages):
        try:
            for line in process.stdout:
                try:
                    messages.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            messages.put(None)

    def _receive(self, seconds):
        try:
            result = self.messages.get(timeout=seconds)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError('Offline speech worker exceeded its deadline') from exc
        if result is None:
            self.close()
            raise RuntimeError('Offline speech worker exited; inspect offline_worker.log')
        return result

    def warmup(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        if not self.python.is_file():
            raise FileNotFoundError('Run scripts/setup_offline_speech.py first')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.messages = queue.Queue()
        self.log = (self.output_dir / 'offline_worker.log').open('a', encoding='utf-8')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_SYMLINKS_WARNING='1')
        self.process = subprocess.Popen(
            [str(self.python), '-I', '-X', 'utf8', '-u', str(self.root / 'scripts/offline_tts_worker.py'),
             '--output-dir', str(self.output_dir)],
            cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding='utf-8', bufsize=1,
        )
        threading.Thread(target=self._read_output, args=(self.process, self.messages), daemon=True).start()
        result = self._receive(self.startup_seconds)
        if result.get('ready') is not True:
            self.close()
            raise RuntimeError('Offline speech worker did not initialize')

    def synthesize_wav(self, text, target):
        self.warmup()
        target = Path(target).resolve()
        if target.parent != self.output_dir or target.suffix != '.wav':
            raise ValueError('Unexpected offline speech output path')
        self.process.stdin.write(json.dumps({'text': text, 'wav_path': str(target)}) + '\n')
        self.process.stdin.flush()
        result = self._receive(self.timeout_seconds)
        if result.get('error'):
            raise RuntimeError(result['error'])
        if Path(result.get('wav_path', '')).resolve() != target or not target.is_file() or target.stat().st_size <= 44:
            raise RuntimeError('Offline speech returned no WAV')

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                try:
                    import psutil
                    owned = psutil.Process(process.pid)
                    for child in owned.children(recursive=True):
                        child.terminate()
                    owned.terminate()
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process.stdin:
                process.stdin.close()
        if self.log is not None:
            self.log.close()
            self.log = None
