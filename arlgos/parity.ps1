# =============================================================================
#  parity.ps1  --  T9's cross-environment check: the fork's corridor env vs ARLGOS's SumoEnv
#
#  Runs arlgos\corridor_parity.py on both sides (3 seeds x fixed-time 60 s and a fixed
#  keep/switch sequence) and compares them: demand, light strings every tick, camera
#  counts, AQL series, T7 metrics.  Writes results\T9\parity.md.  About 5 minutes.
#  -ForkPython picks the fork side's env: unicorn (traci) or unicorn_libsumo with
#  -Libsumo (Decision 7's equality check).  The ARLGOS repo must sit next to the fork.
#  Usage:   .\arlgos\parity.ps1                      # traci, as published
#           .\arlgos\parity.ps1 -Libsumo             # the client the corridor runs use
# =============================================================================
param(
    [string]$ForkPython = "",
    [string]$ArlgosPython = "C:\Users\DELL\miniconda3\envs\arlgos_sumo\python.exe",
    [switch]$Libsumo
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $ForkPython) {
    $ForkPython = $(if ($Libsumo) { "C:\Users\DELL\miniconda3\envs\unicorn_libsumo\python.exe" }
                    else { "C:\Users\DELL\miniconda3\envs\unicorn\python.exe" })
}
$env:PYTHONPATH = ''
$env:PYTHONIOENCODING = 'utf-8'
$env:UNICORN_LIBSUMO = $(if ($Libsumo) { '1' } else { '' })
& $ForkPython -u arlgos\corridor_parity.py fork
if ($LASTEXITCODE -ne 0) { throw "fork side failed" }
$env:UNICORN_LIBSUMO = ''
& $ArlgosPython -u arlgos\corridor_parity.py arlgos
if ($LASTEXITCODE -ne 0) { throw "ARLGOS side failed" }
& $ArlgosPython -u arlgos\corridor_parity.py compare
if ($LASTEXITCODE -ne 0) { throw "parity FAILED: see results\T9\parity.md" }
