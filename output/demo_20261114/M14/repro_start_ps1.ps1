param([string]$Mode = 'cold', [int]$Runs = 3)
# M14 repro: start the app with start.ps1 (unchanged), POST the same image N times, then stop the backend.
# Mode 'cold' = ITAS_WARMUP unset; mode 'warm' = ITAS_WARMUP=1 set in the environment before start.ps1.
$ErrorActionPreference = 'Continue'
$root = 'D:\DSP391m\project\itas-mm-scaffold'
$out = Join-Path $root "output\demo_20261114\M14\$Mode"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$photo = Join-Path $root 'output\demo_20261114\M12\e2e_final\photo_61s.jpg'
if ($Mode -eq 'warm') { $env:ITAS_WARMUP = '1' } else { Remove-Item Env:ITAS_WARMUP -ErrorAction SilentlyContinue }
"ITAS_WARMUP at launch = [$env:ITAS_WARMUP]" | Out-File (Join-Path $out 'startup.txt') -Encoding utf8

$t0 = Get-Date
$p = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'start.ps1')) -WorkingDirectory $root -RedirectStandardOutput (Join-Path $out 'start_stdout.log') -RedirectStandardError (Join-Path $out 'start_stderr.log') -WindowStyle Hidden -PassThru
$ready = $false
for ($i = 0; $i -lt 900; $i++) {
  try { if ((Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ready = $true; break } } catch {}
  Start-Sleep -Seconds 1
}
$startSec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
"ready=$ready start_seconds=$startSec" | Out-File (Join-Path $out 'startup.txt') -Encoding utf8 -Append

if ($ready) {
  for ($r = 1; $r -le $Runs; $r++) {
    $t = Get-Date
    $code = & curl.exe -s -o (Join-Path $out "run$r.json") -w '%{http_code}' -F "image=@$photo" -F latitude=10.7769 -F longitude=106.7009 -F synthesize_audio=false -F annotate=true -F auxiliary_sign_review=false 'http://127.0.0.1:8000/pipeline/run'
    $sec = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    "run=$r http=$code seconds=$sec" | Out-File (Join-Path $out 'runs.txt') -Encoding utf8 -Append
  }
}

# Stop only the uvicorn python process that owns port 8000 (started by start.ps1 above).
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conn) {
  $pr = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
  if ($pr -and $pr.ProcessName -like 'python*') { Stop-Process -Id $pr.Id -Force; "stopped pid=$($pr.Id)" | Out-File (Join-Path $out 'runs.txt') -Encoding utf8 -Append }
}
if (-not $ready) { "backend never became ready" | Out-File (Join-Path $out 'runs.txt') -Encoding utf8 -Append }
"done" | Out-File (Join-Path $out 'done.txt') -Encoding utf8
