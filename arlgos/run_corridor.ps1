# =============================================================================
#  run_corridor.ps1  --  one Unicorn run on the ARLGOS corridor (T9/T10), start to finish
#
#  1. trains Unicorn as published (driver_unicorn.py through arlgos\run_unicorn.ps1:
#     paper hyperparameters, 6 workers, 1500 episodes = 1499 PPO updates, a checkpoint
#     every 50) on maps\corridor_3 (env\corridor.py), with -Sensing full | camera;
#  2. picks the checkpoint on the 20 validation seeds and tests it once on 1042-1061
#     (arlgos\eval_corridor.py), greedy with SUMO's T7 files, and sampled.
#
#  SUMO client: libsumo (env unicorn_libsumo, UNICORN_LIBSUMO=1; Decision 7, checked by
#  arlgos\corridor_parity.py).  -Traci uses the published traci instead (any env).
#
#  Safe to re-run: a finished run (train.done) skips training; a stopped run resumes from
#  model\checkpoint.pkl (run_unicorn.ps1 -Resume); evaluation skips what is complete.
#
#  Output:  Train_MATSC\corridor3_<sensing>_seed<n>\{model, train, config.json, eval\}
#           logs\corridor3_<sensing>_seed<n>.log, logs\..._eval.log
#  Usage:   .\arlgos\run_corridor.ps1 -Sensing full -Seed 1
#           .\arlgos\run_corridor.ps1 -Sensing camera -Seed 1 -Episodes 3   # smoke: 2 updates
# =============================================================================
param(
    [ValidateSet('full', 'camera')][string]$Sensing = 'full',
    [int]$Seed = 1,
    [int]$Episodes = 1500,
    [int]$Workers = 6,
    [int]$EvalJobs = 6,
    [string]$Name = "",
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn_libsumo\python.exe",
    [switch]$Traci,
    [switch]$NoEval
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Name) { $Name = "corridor3_{0}_seed{1}" -f $Sensing, $Seed }
$run = Join-Path "Train_MATSC" $Name
$env:PYTHONPATH = ''
$env:PYTHONIOENCODING = 'utf-8'
$env:UNICORN_SENSING = $Sensing
$env:UNICORN_LIBSUMO = $(if ($Traci) { '' } else { '1' })
function Stamp { Get-Date -Format "yyyy-MM-dd HH:mm" }

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction SilentlyContinue
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
try {
    $done = Join-Path $run "train.done"
    if (Test-Path $done) {
        Write-Host ("[{0}] {1}: trained already" -f (Stamp), $Name)
    } else {
        $resume = Test-Path (Join-Path $run "model\checkpoint.pkl")
        if ((Test-Path $run) -and -not $resume) {
            $aside = "{0}_aborted_{1}" -f $run, (Get-Date -Format yyyyMMddHHmm)
            Move-Item $run $aside
            Write-Host "moved an unstarted earlier attempt to $aside"
        }
        Write-Host ("[{0}] {1}: train ({2}, {3}, client {4})" -f (Stamp), $Name, $Sensing,
                    $(if ($resume) { 'resume' } else { 'new' }), $(if ($Traci) { 'traci' } else { 'libsumo' }))
        $a = @{ Episodes = $Episodes; Seed = $Seed; Workers = $Workers; Net = 'corridor_3'; Name = $Name; Python = $Python }
        if ($resume) { & (Join-Path $PSScriptRoot "run_unicorn.ps1") @a -Resume }
        else { & (Join-Path $PSScriptRoot "run_unicorn.ps1") @a }
        Set-Content -Path $done -Value (Stamp)
    }
    if (-not $NoEval) {
        Write-Host ("[{0}] {1}: validation + test" -f (Stamp), $Name)
        $log = Join-Path "logs" "${Name}_eval.log"
        & $Python -u arlgos\run_logged.py $log -- $Python -u arlgos\eval_corridor.py --run $run --jobs $EvalJobs
        if ($LASTEXITCODE -ne 0) { throw "eval_corridor.py exited $LASTEXITCODE; see $log" }
    }
    Write-Host ("[{0}] Done: {1}" -f (Stamp), $run)
}
finally {
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
