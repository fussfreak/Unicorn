# =============================================================================
#  run_fenglin_unicorn.ps1  --  T15F: Unicorn on Fenglin (shaoxing_network_7), for the lab PC
#
#  For each seed, one after another (a run uses 6 workers + the learner):
#    1. train Unicorn as published (driver_unicorn.py via arlgos\run_unicorn.ps1: paper
#       hyperparameters, 6 workers, 1500 episodes = 1499 PPO updates, a checkpoint every 50);
#    2. choose the checkpoint on validation seeds 5042-5046 and test it once on Unicorn's 10 test
#       seeds 1000..1900, greedy and sampled (arlgos\eval_unicorn.py): the rule ARLGOS-v2 follows.
#  Then the head-to-head table (arlgos\fenglin_report.py) with whatever runs exist.
#
#  Safe to re-run after a reboot or crash: finished runs are skipped, a stopped run resumes from
#  model\checkpoint.pkl, evaluation skips what is complete. A crashed run is retried up to
#  -Retries times (Ray processes it left behind are ended first), as T10's queue does.
#
#  Fenglin's trajectory file fcd.xml is off (maps\shaoxing_network_7\build_file.py; checked
#  identical by arlgos\fenglin_check.py). -Demand heldout trains and tests on held-out traffic
#  days instead of the published fixed route file (runs are then named unicorn_heldout_...).
#
#  Output:  Train_MATSC\unicorn_shaoxing_network_7_seed<k>\{model, train, config.json, eval\}
#           logs\T15F_unicorn_queue.log;  results\T15F\fenglin_h2h.md  <- push this back
#  Usage:   .\arlgos\run_fenglin_unicorn.ps1 -Python <env python>          # seeds 1, 2, 3
#           .\arlgos\run_fenglin_unicorn.ps1 -Seeds 1 -Episodes 2 -Workers 2 -Prefix smoke_unicorn -SmokeEval
#                                                    # pipeline check: 1 update, short evaluation
# =============================================================================
param(
    [int[]]$Seeds = @(1, 2, 3),
    [int]$Episodes = 1500,
    [int]$Workers = 6,
    [int]$EvalJobs = 6,
    [ValidateSet('published', 'heldout')][string]$Demand = 'published',
    [string]$Prefix = "",
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn_libsumo\python.exe",
    [int]$Retries = 3,
    [int]$RetryWaitSec = 60,
    [switch]$Traci,
    [switch]$NoEval,
    [switch]$SmokeEval
)
$ErrorActionPreference = 'Stop'
$fork = Split-Path $PSScriptRoot -Parent
Set-Location $fork
$net = 'shaoxing_network_7'
if (-not $Prefix) { $Prefix = $(if ($Demand -eq 'heldout') { 'unicorn_heldout' } else { 'unicorn' }) }
New-Item -ItemType Directory -Force "logs" | Out-Null
$queueLog = "logs\T15F_unicorn_queue.log"
function Say($m) { $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m; Write-Host $line; Add-Content $queueLog $line }

$env:PYTHONPATH = ''
$env:PYTHONIOENCODING = 'utf-8'
$env:UNICORN_LIBSUMO = $(if ($Traci) { '' } else { '1' })
$env:UNICORN_DEMAND = $Demand
$env:UNICORN_FCD = '0'

function Clear-Ray {
    $envDir = (Split-Path $Python -Parent).TrimEnd('\') + '\'
    $left = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($envDir, [StringComparison]::OrdinalIgnoreCase) -and
        $_.CommandLine -notmatch 'driver_arlgos\.py|watch_' })
    foreach ($p in $left) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    return $left.Count
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction SilentlyContinue
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
try {
    foreach ($seed in $Seeds) {
        $name = "{0}_{1}_seed{2}" -f $Prefix, $net, $seed
        $run = Join-Path "Train_MATSC" $name
        for ($attempt = 0; ; $attempt++) {
            try {
                $done = Join-Path $run "train.done"
                if (-not (Test-Path $done)) {
                    $resume = Test-Path (Join-Path $run "model\checkpoint.pkl")
                    if ((Test-Path $run) -and -not $resume) {
                        $aside = "{0}_aborted_{1}" -f $run, (Get-Date -Format yyyyMMddHHmm)
                        Move-Item $run $aside
                        Say "moved an unstarted earlier attempt to $aside"
                    }
                    Say ("train {0} ({1}, demand {2}){3}" -f $name, $(if ($resume) { 'resume' } else { 'new' }), $Demand,
                         $(if ($attempt) { " retry $attempt of $Retries" } else { '' }))
                    $a = @{ Episodes = $Episodes; Seed = $seed; Workers = $Workers; Net = $net; Name = $name; Python = $Python }
                    if ($resume) { & (Join-Path $PSScriptRoot "run_unicorn.ps1") @a -Resume }
                    else { & (Join-Path $PSScriptRoot "run_unicorn.ps1") @a }
                    Set-Content -Path $done -Value (Get-Date -Format "yyyy-MM-dd HH:mm")
                    Say "trained $name"
                }
                if (-not $NoEval) {
                    Say "validate + test $name"
                    $log = Join-Path "logs" "${name}_eval.log"
                    $evalArgs = @('-u', 'arlgos\eval_unicorn.py', '--run', $run, '--jobs', "$EvalJobs")
                    if ($SmokeEval) { $evalArgs += '--smoke' }
                    & $Python -u arlgos\run_logged.py $log -- $Python @evalArgs
                    if ($LASTEXITCODE -ne 0) { throw "eval_unicorn.py exited $LASTEXITCODE; see $log" }
                    Say "tested $name"
                }
                break
            } catch {
                if ($attempt -ge $Retries) { Say "FAILED $name after $Retries retries: $_"; throw }
                Say "crashed $name ($_); retrying in $RetryWaitSec s from its last saved update"
                Say ("  ended {0} leftover process(es)" -f (Clear-Ray))
                Start-Sleep -Seconds $RetryWaitSec
            }
        }
    }
    if (-not $NoEval -and -not $SmokeEval) {
        & $Python arlgos\fenglin_report.py --demand $Demand
        Say "report: results\T15F\fenglin_h2h$(if ($Demand -eq 'heldout') { '_heldout' } else { '' }).md"
    }
    Say "Unicorn Fenglin queue finished"
}
finally {
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
