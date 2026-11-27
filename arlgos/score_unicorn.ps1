# =============================================================================
#  score_unicorn.ps1  --  T15A: score the supervisor's Unicorn runs with ARLGOS-v2's rule, on the lab PC
#
#    1. arlgos\import_unicorn_runs.py --src <folder>: copies each original-code Unicorn run
#       (model\checkpoint{N}.pkl + config.json) into Train_MATSC\unicornsup_<net>_seed<k>\
#       (unicornsupmulti_... for a jointly trained run). The supervisor's folders are only read.
#    2. arlgos\eval_unicorn.py --run <each imported run>: picks the checkpoint on validation seeds
#       5042-5046, then tests it once on Unicorn's 10 test seeds, greedy and sampled -- the rule
#       ARLGOS-v2's runs follow. One run at a time, -Jobs episodes in parallel.
#    3. arlgos\h2h_report.py --net <net> after each run, and --all at the end (results\h2h\).
#
#  Runs unattended: starts itself in the background and returns; one progress window; Windows kept
#  awake; safe to re-run (finished parts are skipped); -Stop ends it.
#
#  Usage (from the fork, e.g. C:\arlgos_lab\Unicorn):
#    .\arlgos\score_unicorn.ps1 -Src "D:\<supervisor>\Unicorn\Train_MATSC" -Python <arlgos_unicorn_libsumo python>
#    .\arlgos\score_unicorn.ps1 -Src ... -DryRun          # list the runs it would import; copy nothing
#    .\arlgos\score_unicorn.ps1 -Stop
# =============================================================================
param(
    [string[]]$Src = @(),
    [string]$Python = '',
    [int]$Jobs = 6,
    [string[]]$Nets = @(),
    [switch]$Smoke,                  # pipeline check: eval_unicorn.py --smoke (short episodes, eval_smoke\; not a result)
    [switch]$NoWindow,
    [switch]$Stop,
    [switch]$DryRun,
    [switch]$Detached
)
$ErrorActionPreference = 'Stop'
$fork = Split-Path $PSScriptRoot -Parent
Set-Location $fork
$qdir = "logs\labpc_queue"
New-Item -ItemType Directory -Force $qdir | Out-Null
$log = Join-Path $qdir "score_unicorn.log"
function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m
    Write-Host $line
    try { Add-Content $log $line } catch { }
}
function Q($s) { "'" + ([string]$s).Replace("'", "''") + "'" }
function FindEnv($name) {
    $roots = @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniconda3",
               "C:\ProgramData\miniconda3", "C:\ProgramData\Anaconda3", "C:\miniconda3")
    foreach ($r in $roots) { $p = Join-Path $r "envs\$name\python.exe"; if (Test-Path $p) { return $p } }
    return ''
}
function OurProcs($pattern, $names) {
    $filter = ($names | ForEach-Object { "Name = '$_'" }) -join ' OR '
    @(Get-CimInstance Win32_Process -Filter $filter |
      Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $pattern })
}

if ($Stop) {
    $main = @(OurProcs 'score_unicorn\.ps1' @('powershell.exe', 'pwsh.exe'))
    foreach ($p in $main) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    $evals = @(OurProcs 'eval_unicorn\.py|import_unicorn_runs\.py' @('python.exe'))
    $children = @()
    foreach ($e in $evals) {                                    # eval_unicorn.py's worker pool
        $children += @(OurProcs ("parent_pid=" + $e.ProcessId + "\b") @('python.exe'))
        Stop-Process -Id $e.ProcessId -Force -ErrorAction SilentlyContinue
    }
    foreach ($c in $children) { Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue }
    Say ("stopped: {0} scorer, {1} python process(es), {2} worker(s); re-run to resume" -f $main.Count, $evals.Count, $children.Count)
    return
}

if (-not $Python) { $Python = FindEnv 'arlgos_unicorn_libsumo' }
if (-not $Python -or -not (Test-Path $Python)) { throw "-Python: the arlgos_unicorn_libsumo env's python.exe was not found; pass -Python <path>" }
if (-not $Src.Count) { throw "-Src: the folder holding the supervisor's Unicorn runs (their Train_MATSC)" }
foreach ($s in $Src) { if (-not (Test-Path $s)) { throw "-Src $s : no such folder" } }
$env:PYTHONPATH = ''; $env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
$env:UNICORN_FCD = '0'; $env:UNICORN_DEMAND = 'published'
$srcArgs = @(); foreach ($s in $Src) { $srcArgs += @('--src', $s) }

if ($DryRun) {
    & $Python arlgos\import_unicorn_runs.py @srcArgs --dry_run
    return
}

if (-not $Detached) {
    $running = @(OurProcs 'score_unicorn\.ps1' @('powershell.exe', 'pwsh.exe'))
    if ($running.Count) { Say "a scorer is already running (pid $($running[0].ProcessId)); use -Stop first"; return }
    $cmd = "& {0} -Detached -Src {1} -Python {2} -Jobs {3}" -f (Q $PSCommandPath), (($Src | ForEach-Object { Q $_ }) -join ','),
        (Q $Python), $Jobs
    if ($Nets.Count) { $cmd += " -Nets " + (($Nets | ForEach-Object { Q $_ }) -join ',') }
    if ($Smoke) { $cmd += " -Smoke" }
    if ($NoWindow) { $cmd += " -NoWindow" }
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd) | Out-Null
    Write-Host "Scoring started in the background. Log: $log"
    Write-Host "Stop it with: .\arlgos\score_unicorn.ps1 -Stop    (re-run the same command to resume)"
    return
}

$env:SUMO_HOME = (& $Python -c "import sumo; print(sumo.SUMO_HOME)").Trim()
$hasLibsumo = (& $Python -c "import importlib.util as u; print(int(u.find_spec('libsumo') is not None))").Trim()
$env:UNICORN_LIBSUMO = $(if ($hasLibsumo -eq '1') { '1' } else { '' })
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
try {
    if (-not $NoWindow) {
        $watch = "Set-Location '$fork'; `$host.UI.RawUI.WindowTitle = 'Unicorn scoring (closing this window stops nothing)'; " +
                 "while (`$true) { Clear-Host; Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; Write-Host ''; Get-Content '$log' -Tail 8 -ErrorAction SilentlyContinue; " +
                 "Get-ChildItem 'logs\score_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { " +
                 "Write-Host ''; Write-Host `$_.Name -ForegroundColor Cyan; Get-Content `$_.FullName -Tail 4 }; Start-Sleep 30 }"
        Start-Process powershell -ArgumentList "-NoProfile", "-Command", $watch | Out-Null
    }
    Say ("import from {0} (client {1})" -f ($Src -join ', '), $(if ($env:UNICORN_LIBSUMO) { 'libsumo' } else { 'traci' }))
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'      # PS 5.1: stderr lines are not errors
    # Add-Content, not Tee-Object: Windows PowerShell 5.1's Tee-Object -Append writes UTF-16 into this ANSI log
    & $Python -u arlgos\import_unicorn_runs.py @srcArgs 2>&1 | ForEach-Object { "$_" } | Add-Content $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $eap
    if ($code -ne 0) { throw "import_unicorn_runs.py exited $code; see $log" }

    $runs = @(Get-ChildItem Train_MATSC -Directory | Where-Object { $_.Name -match '^unicornsup(multi)?_(.+)_seed\d+$' })
    if ($Nets.Count) { $runs = @($runs | Where-Object { $Nets -contains ($_.Name -replace '^unicornsup(multi)?_(.+)_seed\d+$', '$2') }) }
    Say "$($runs.Count) imported run(s) to score"
    $failed = @()
    $evalDir = $(if ($Smoke) { 'eval_smoke' } else { 'eval' })
    foreach ($r in $runs) {
        $net = $r.Name -replace '^unicornsup(multi)?_(.+)_seed\d+$', '$2'
        if ((Test-Path (Join-Path $r.FullName "$evalDir\test.json")) -and (Test-Path (Join-Path $r.FullName "$evalDir\test_sampled.json"))) { continue }
        Say ("score {0}{1}" -f $r.Name, $(if ($Smoke) { ' (pipeline check)' } else { '' }))
        $t0 = Get-Date
        $e = @('-u', 'arlgos\eval_unicorn.py', '--run', (Join-Path 'Train_MATSC' $r.Name), '--jobs', $Jobs)
        if ($Smoke) { $e += '--smoke' }
        $ErrorActionPreference = 'Continue'
        & $Python @e 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 ("logs\score_{0}.log" -f $r.Name)
        $code = $LASTEXITCODE
        if (-not $Smoke) {
            & $Python arlgos\h2h_report.py --net $net 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $qdir "h2h_$net.log")
        }
        $ErrorActionPreference = $eap
        if ($code -ne 0) { $failed += $r.Name; Say "  FAILED (exit $code): see logs\score_$($r.Name).log"; continue }
        Say ("  done in {0:N0} min: {1}" -f ((Get-Date) - $t0).TotalMinutes, (Join-Path $r.Name "$evalDir\test.md"))
    }
    if (-not $Smoke) {
        $ErrorActionPreference = 'Continue'
        & $Python arlgos\h2h_report.py --all 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $qdir "h2h_all.log")
        $ErrorActionPreference = $eap
    }
    Say ("scoring finished{0}; overview: results\h2h\overview.md" -f $(if ($failed.Count) { " (FAILED: $($failed -join ', '))" } else { '' }))
}
finally {
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
