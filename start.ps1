param([int]$Port = 8000)
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot\.runtime-packages;$PSScriptRoot"
python -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
