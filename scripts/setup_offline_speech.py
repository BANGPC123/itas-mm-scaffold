"""Install pinned minimal ONNX speech in its own venv, then cache its model."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = '3.8.3'


def main():
    env = ROOT / '.tts-venv'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        subprocess.run([sys.executable, '-I', '-m', 'venv', str(env)], check=True)
    subprocess.run([str(python), '-I', '-m', 'pip', 'install', f'vieneu=={VERSION}'], check=True)
    # Download on setup. Normal serving sets HF_HUB_OFFLINE=1 and needs no network.
    code = ("from vieneu import Vieneu; from pathlib import Path; "
            "m=Vieneu(backend='onnx',precision='int8'); "
            "a=m.infer('Bộ đọc tiếng Việt đã sẵn sàng.',voice='Hải Đăng'); "
            "Path('output').mkdir(exist_ok=True); m.save(a,'output/offline_speech_setup.wav')")
    subprocess.run([str(python), '-I', '-c', code], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
