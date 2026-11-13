param([int]$Port = 8000)
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot\.runtime-packages;$PSScriptRoot"
# User-site packages break anyio/fastapi on this machine; the offline TTS model cache lives in the repo.
$env:PYTHONNOUSERSITE = '1'
$env:HF_HUB_DISABLE_SYMLINKS = '1'
# Warm up models before the server reports healthy (M14: avoids ~33 s on the first image upload).
$env:ITAS_WARMUP = '1'
$env:HF_HOME = Join-Path $PSScriptRoot 'models\tts_hf_home'
$python = 'D:\miniconda\envs\itas-mm\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
& $python -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
