# ITAS-MM capstone launcher, only Web/Desktop MP4 demo; no physical camera/GPS.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$python = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path (Join-Path $root 'frontend/dist/index.html'))) {
  Write-Host 'Frontend build missing; run: npm --prefix frontend run build'
  exit 2
}
$ok=$false
try {$ok=((Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200)}catch{}
if(-not $ok) {
  $env:PYTHONPATH="$root\.runtime-packages;$root"
  $env:ITAS_WARMUP='1'
  $env:PYTHONIOENCODING='utf-8'
  $server=Start-Process -FilePath $python -WorkingDirectory $root -ArgumentList @('-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000') -RedirectStandardOutput (Join-Path $root 'output/capstone_api.stdout.log') -RedirectStandardError (Join-Path $root 'output/capstone_api.stderr.log') -WindowStyle Hidden -PassThru
  Write-Host ('Backend PID = '+$server.Id)
  for($i=0;$i -lt 240;$i++) {
     Start-Sleep -Seconds 1
     try {$ok=((Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200);if($ok){break}}catch{}
  }
}
if(-not $ok){throw 'Cannot start backend; inspect output/capstone_api.stderr.log'}
Write-Host 'ITAS-MM desktop demo is online at http://127.0.0.1:8000'
Write-Host 'Full test video: http://127.0.0.1:8000/pipeline/capstone-video/user-complete'
Start-Process 'http://127.0.0.1:8000'
