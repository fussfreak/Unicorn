# =============================================================================
#  run_T10.ps1  --  T10's six Unicorn runs on the ARLGOS corridor, one after another
#
#  Order: full seed 1, camera seed 1, full 2, camera 2, full 3, camera 3, so that each
#  finished pair already gives a full-vs-camera comparison.  Each run is
#  arlgos\run_corridor.ps1 (train 1499 updates, validate, test once).  One run at a time:
#  a run uses 6 workers + the learner, i.e. the whole laptop.
#
#  Safe to re-run after a reboot or a crash: finished runs are skipped, a stopped run
#  resumes from its last update.  Log: logs\T10_queue.log
#
#  Retries (added 2026-11-08): a run that crashes is retried up to -Retries times.  Before
#  each retry the Ray processes it left behind are ended (only those of -Python's env;
#  a sumo-gui viewer, watch_corridor.py, is kept) and the queue waits -RetryWaitSec.  A
#  retry resumes from the last saved update, so it repeats no training.  Why: on
#  2026-11-08 Ray's dashboard agent found its random port taken (WSA 10048) and the run
#  stopped, which left the laptop idle until someone looked.
#
#  Usage:  .\arlgos\run_T10.ps1                 # everything
#          .\arlgos\run_T10.ps1 -Seeds 1        # just the first pair
#  (-Runner, -Log and -NoCleanup exist for arlgos\check_corridor.py's test of the retries.)
# =============================================================================
param(
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$Sensing = @('full', 'camera'),
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn_libsumo\python.exe",
    [int]$Retries = 3,
    [int]$RetryWaitSec = 60,
    [string]$Runner = "",
    [string]$Log = "logs\T10_queue.log",
    [switch]$NoCleanup
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Runner) { $Runner = Join-Path $PSScriptRoot "run_corridor.ps1" }
New-Item -ItemType Directory -Force (Split-Path $Log -Parent) | Out-Null
function Say($m) { $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m; Write-Host $line; Add-Content $Log $line }

function Clear-Ray {
    # Processes still running from -Python's env after a crash: raylet, gcs_server, Ray's
    # workers and dashboard agent all live under that env's folder.  Nothing else is touched.
    $envDir = (Split-Path $Python -Parent).TrimEnd('\') + '\'
    $left = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($envDir, [StringComparison]::OrdinalIgnoreCase) -and
        $_.CommandLine -notmatch 'watch_corridor\.py' })
    foreach ($p in $left) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    return $left.Count
}

foreach ($seed in $Seeds) {
    foreach ($s in $Sensing) {
        $name = "corridor3_${s}_seed${seed}"
        for ($attempt = 0; ; $attempt++) {
            Say ("start {0}{1}" -f $name, $(if ($attempt) { " (retry $attempt of $Retries)" } else { '' }))
            try {
                & $Runner -Sensing $s -Seed $seed -Python $Python
                Say "done  $name"
                break
            } catch {
                if ($attempt -ge $Retries) {
                    Say "FAILED $name after $Retries retries: $_"
                    throw
                }
                Say "crashed $name ($_); retrying in $RetryWaitSec s from its last saved update"
                if (-not $NoCleanup) { Say ("  ended {0} leftover process(es) of {1}" -f (Clear-Ray), (Split-Path (Split-Path $Python -Parent) -Leaf)) }
                Start-Sleep -Seconds $RetryWaitSec
            }
        }
    }
}
Say "T10 queue finished"
