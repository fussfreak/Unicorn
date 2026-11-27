# =============================================================================
#  watch_arlgos.ps1  --  watch an ARLGOS-v2 checkpoint drive a benchmark network in sumo-gui
#
#  Plays one hour of demo seed 9042 (not a training, validation or test seed) with the run's
#  best-on-validation policy (model\best_val.pt), or with -Latest the newest ckpt_<ep>.pt.
#  Read-only: it writes nothing into the run and does not disturb a training in progress.
#  Uses the traci client (sumo-gui cannot run under libsumo), i.e. the 'unicorn' env.
#
#  Usage:  .\arlgos\watch_arlgos.ps1 -Net shaoxing_network_7 -Seed 1
#          .\arlgos\watch_arlgos.ps1 -Net shaoxing_network_7 -Seed 2 -Latest
#  In sumo-gui: press the green Play button; set "Delay (ms)" to about 50 to see the cars move.
# =============================================================================
param(
    [string]$Net = "shaoxing_network_7",
    [int]$Seed = 1,
    [string]$Prefix = "arlgos",
    [switch]$Latest,
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn\python.exe"
)
$ErrorActionPreference = 'Stop'
$fork = Split-Path $PSScriptRoot -Parent
Set-Location $fork
$run = Join-Path "Train_MATSC" ("{0}_{1}_seed{2}" -f $Prefix, $Net, $Seed)
if (-not (Test-Path $run)) { throw "$run not found" }
$ckpt = "best_val.pt"
if ($Latest) {
    $last = Get-ChildItem (Join-Path $run "model") -Filter "ckpt_*.pt" | Sort-Object Name | Select-Object -Last 1
    if (-not $last) { throw "no ckpt_*.pt yet in $run\model (the first is saved after 50 episodes)" }
    $ckpt = $last.Name
} elseif (-not (Test-Path (Join-Path $run "model\best_val.pt"))) {
    throw "no best_val.pt yet in $run\model (the first validation is after 50 episodes); try -Latest later"
}
$env:PYTHONPATH = ''
$env:PYTHONIOENCODING = 'utf-8'
$env:UNICORN_LIBSUMO = ''
$env:UNICORN_FCD = '0'
$cfg = Get-Content (Join-Path $run "config.json") -Raw | ConvertFrom-Json
$env:UNICORN_DEMAND = $(if ($cfg.demand) { $cfg.demand } else { 'published' })
$env:SUMO_HOME = (& $Python -c "import sumo; print(sumo.SUMO_HOME)").Trim()
Write-Host "watching $run\model\$ckpt (demand $($env:UNICORN_DEMAND)) in sumo-gui ..."
& $Python -u arlgos\eval_arlgos.py --run $run --checkpoint $ckpt --watch
