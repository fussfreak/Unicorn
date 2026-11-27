# =============================================================================
#  run_labpc_queue.ps1  --  T15A: ARLGOS-v2 on every network of the Unicorn paper, on the lab PC
#
#  Works through -Nets in order, network by network (user decision 2026-11-25). For each network:
#    1. run_arlgos.ps1: trains the seeds side by side (one CPU thread each), tests each run's
#       best-on-validation checkpoint once on Unicorn's 10 test seeds, writes its table
#       (results\arlgos_v2\<Prefix>_<net>.md, results\paper_compare\<net>_<Prefix>.md);
#    2. nonrl_probe.ps1: fixed-time, greedy and max-pressure in the same simulator, on this
#       machine (results\nonrl_probe_labpc\<net>.*); skipped with -SkipNonRL;
#    3. h2h_report.py --net: ARLGOS-v2 vs the supervisor's Unicorn rerun (results\h2h\<net>.md;
#       "no comparison yet" until arlgos\score_unicorn.ps1 has scored that network's runs).
#  At the end: h2h_report.py --all -> results\h2h\overview.md.
#
#  Runs unattended:
#    * It starts itself in the background and returns; closing your window stops nothing.
#    * One progress window shows the queue log and the newest run logs (closing it stops nothing).
#    * Windows is kept awake while it runs.
#    * -Parallel K: up to K networks at once (each uses one thread per seed, so 3 by default).
#    * Safe to re-run after a crash, a reboot or -Stop: finished networks are skipped, stopped
#      trainings resume from their last saved state (at most 5 episodes lost).
#    * -Stop ends the queue, then its network workers, then their launchers and python, in that
#      order (a launcher left alive restarts its seed). The supervisor's processes are not touched.
#
#  Usage (from the fork, e.g. C:\arlgos_lab\Unicorn):
#    .\arlgos\run_labpc_queue.ps1 -Python <arlgos_unicorn_libsumo python> -NonRLPython <arlgos_unicorn python> -Parallel 2
#    .\arlgos\run_labpc_queue.ps1 -Nets cologne_network_8,shaoxing_network_7 ...      # another order / subset
#    .\arlgos\run_labpc_queue.ps1 -DryRun                                             # what it would do
#    .\arlgos\run_labpc_queue.ps1 -Stop
#    pipeline check (minutes): -Nets cologne_network_8 -Seeds 1 -Episodes 3 -SimSeconds 300 -Prefix smoke_q -SkipNonRL
# =============================================================================
param(
    [string[]]$Nets = @('cologne_network_8', 'arterial_network_4_4', 'grid_network_4_4', 'ingolstadt_network_21',
                        'shaoxing_network_7', 'grid_network_5_5', 'monaco_network_30', 'shenzhen_network_29'),
    [int[]]$Seeds = @(1, 2, 3),
    [int]$Episodes = 1500,
    [string]$Prefix = 'arlgos_labpc',
    [string[]]$Extra = @('--val_every', '50'),
    [int]$SimSeconds = 0,
    [string]$Python = '',
    [string]$NonRLPython = '',
    [int]$Parallel = 1,
    [int]$Retries = 2,
    [switch]$SkipNonRL,
    [switch]$NoWindow,
    [switch]$Stop,
    [switch]$DryRun,
    [switch]$Detached,
    [string]$WorkerNet = '',
    [int]$ServerBase = 600
)
$ErrorActionPreference = 'Stop'
$fork = Split-Path $PSScriptRoot -Parent
Set-Location $fork
$qdir = "logs\labpc_queue"
New-Item -ItemType Directory -Force $qdir | Out-Null
$queueLog = Join-Path $qdir "$Prefix.queue.log"
function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m
    Write-Host $line
    try { Add-Content $queueLog $line } catch { }             # a reader holding the file never stops a run
}
function Q($s) { "'" + ([string]$s).Replace("'", "''") + "'" }
function Marker($net) { Join-Path $qdir ("{0}_{1}.done" -f $Prefix, $net) }
function FindEnv($name) {
    # this env's python.exe under the usual conda roots ('' if none)
    $roots = @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniconda3",
               "C:\ProgramData\miniconda3", "C:\ProgramData\Anaconda3", "C:\miniconda3")
    foreach ($r in $roots) { $p = Join-Path $r "envs\$name\python.exe"; if (Test-Path $p) { return $p } }
    return ''
}
if (-not $Python) { $Python = FindEnv 'arlgos_unicorn_libsumo' }
if (-not $NonRLPython) { $NonRLPython = FindEnv 'arlgos_unicorn' }

function OurProcs($pattern, $names) {
    $filter = ($names | ForEach-Object { "Name = '$_'" }) -join ' OR '
    @(Get-CimInstance Win32_Process -Filter $filter |
      Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $pattern })
}

# ---- -Stop -------------------------------------------------------------------------------------
if ($Stop) {
    $ps = @('powershell.exe', 'pwsh.exe')
    $main = @(OurProcs 'run_labpc_queue\.ps1' $ps | Where-Object { $_.CommandLine -notmatch '-WorkerNet' })
    foreach ($p in $main) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    $workers = @(OurProcs 'run_labpc_queue\.ps1.*-WorkerNet' $ps)
    foreach ($p in $workers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    $launchers = @(OurProcs 'run_arlgos\.ps1|nonrl_probe\.ps1' $ps)
    foreach ($p in $launchers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    $py = @(OurProcs 'driver_arlgos\.py|eval_arlgos\.py|h2h_report\.py|paper_compare\.py|nonrl_probe\.py|version_check\.py"?\s+run' @('python.exe'))
    foreach ($p in $py) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Say ("stopped: {0} queue, {1} network worker(s), {2} launcher(s), {3} python process(es); re-run to resume" -f
         $main.Count, $workers.Count, $launchers.Count, $py.Count)
    return
}

if (-not $Python -or -not (Test-Path $Python)) { throw "-Python: the arlgos_unicorn_libsumo env's python.exe was not found; pass -Python <path>" }
if (-not $SkipNonRL -and (-not $NonRLPython -or -not (Test-Path $NonRLPython))) {
    throw "-NonRLPython: the arlgos_unicorn env's python.exe was not found; pass -NonRLPython <path> (or -SkipNonRL)"
}
$h2hArgs = @('arlgos\h2h_report.py', '--arlgos_prefix', $Prefix)
if ($SimSeconds -gt 0) { $h2hArgs += '--include_pipeline_checks' }

# ---- one network (a background worker started by the queue) -------------------------------------
if ($WorkerNet) {
    $net = $WorkerNet
    $env:PYTHONPATH = ''; $env:PYTHONIOENCODING = 'utf-8'
    try {
        Say "$net : ARLGOS-v2 seeds $($Seeds -join ',') ($Episodes episodes)"
        $a = @{ Net = $net; Seeds = $Seeds; Episodes = $Episodes; Prefix = $Prefix; Python = $Python; NoWindow = $true }
        if ($Extra.Count) { $a.Extra = $Extra }
        if ($SimSeconds -gt 0) { $a.SimSeconds = $SimSeconds }
        & (Join-Path $PSScriptRoot 'run_arlgos.ps1') @a
        if (-not $SkipNonRL) {
            Say "$net : non-RL controllers (fixed-time, greedy, max-pressure) on this machine"
            & (Join-Path $PSScriptRoot 'nonrl_probe.ps1') -Net $net -Python $NonRLPython -ServerBase $ServerBase -OutRoot 'results\nonrl_probe_labpc'
        }
        $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'     # PS 5.1: stderr lines are not errors
        & $Python @h2hArgs --net $net 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $qdir "h2h_$net.log")
        $code = $LASTEXITCODE
        $ErrorActionPreference = $eap
        if ($code -ne 0) { throw "h2h_report.py --net $net exited $code (see $qdir\h2h_$net.log)" }
        Set-Content (Marker $net) (Get-Date -Format "yyyy-MM-dd HH:mm")
        Say "$net : done (results\h2h\$net.md)"
        exit 0
    } catch {
        Say "$net : FAILED: $_"
        exit 1
    }
}

# ---- -DryRun -------------------------------------------------------------------------------------
if ($DryRun) {
    "queue {0}: {1} network(s), {2} at a time, seeds {3}, {4} episodes, extra '{5}'{6}" -f $Prefix, $Nets.Count, $Parallel,
        ($Seeds -join ','), $Episodes, ($Extra -join ' '), $(if ($SimSeconds) { ", PIPELINE CHECK ($SimSeconds s episodes)" } else { '' })
    $i = 0
    foreach ($net in $Nets) {
        $state = if (Test-Path (Marker $net)) { 'done, skip' } else {
            $tested = @($Seeds | Where-Object { Test-Path ("Train_MATSC\{0}_{1}_seed{2}\eval\test.json" -f $Prefix, $net, $_) }).Count
            "to do (seeds tested: $tested of $($Seeds.Count))" }
        "{0}. {1}: {2}; then {3}h2h report" -f (++$i), $net, $state, $(if ($SkipNonRL) { '' } else { 'non-RL controllers (servers ' + ($ServerBase + 10 * ($i - 1)) + '+), ' })
    }
    "python: $Python"
    if (-not $SkipNonRL) { "non-RL python: $NonRLPython" }
    return
}

# ---- start in the background (closing the user's window must not stop the queue) -------------------
if (-not $Detached) {
    $running = @(OurProcs 'run_labpc_queue\.ps1' @('powershell.exe', 'pwsh.exe') | Where-Object { $_.CommandLine -notmatch '-WorkerNet' })
    if ($running.Count) { Say "a queue is already running (pid $($running[0].ProcessId)); use -Stop first"; return }
    $cmd = "& {0} -Detached -Nets {1} -Seeds {2} -Episodes {3} -Prefix {4} -SimSeconds {5} -Python {6} -Parallel {7} -Retries {8} -ServerBase {9}" -f
        (Q $PSCommandPath), (($Nets | ForEach-Object { Q $_ }) -join ','), ($Seeds -join ','), $Episodes, (Q $Prefix),
        $SimSeconds, (Q $Python), $Parallel, $Retries, $ServerBase
    if ($Extra.Count) { $cmd += " -Extra " + (($Extra | ForEach-Object { Q $_ }) -join ',') }
    if ($NonRLPython) { $cmd += " -NonRLPython " + (Q $NonRLPython) }
    if ($SkipNonRL) { $cmd += " -SkipNonRL" }
    if ($NoWindow) { $cmd += " -NoWindow" }
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd) | Out-Null
    Write-Host "Queue started in the background. Log: $queueLog"
    Write-Host "Stop it with: .\arlgos\run_labpc_queue.ps1 -Stop    (re-run the same command to resume)"
    return
}

# ---- the queue (detached) -------------------------------------------------------------------------
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
try {
    Say ("queue start: {0}; {1} at a time; seeds {2}; {3} episodes; python {4}" -f ($Nets -join ' > '), $Parallel,
         ($Seeds -join ','), $Episodes, $Python)
    if (-not $NoWindow) {
        $watch = "Set-Location '$fork'; `$host.UI.RawUI.WindowTitle = 'ARLGOS-v2 lab-PC queue (closing this window stops nothing)'; " +
                 "while (`$true) { Clear-Host; Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; Write-Host ''; Get-Content '$queueLog' -Tail 6 -ErrorAction SilentlyContinue; " +
                 "Get-ChildItem 'logs\${Prefix}_*.log','logs\probe_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 4 | ForEach-Object { " +
                 "Write-Host ''; Write-Host `$_.Name -ForegroundColor Cyan; Get-Content `$_.FullName -Tail 2 }; Start-Sleep 30 }"
        Start-Process powershell -ArgumentList "-NoProfile", "-Command", $watch | Out-Null
    }
    $attempts = @{}; $failed = @{}
    foreach ($net in $Nets) { $attempts[$net] = 0 }
    function Worker($net) { @(OurProcs ('run_labpc_queue\.ps1.*-WorkerNet ''?' + [regex]::Escape($net) + '''?(\s|$)') @('powershell.exe', 'pwsh.exe')) }
    while ($true) {
        $pending = @($Nets | Where-Object { -not (Test-Path (Marker $_)) -and -not $failed[$_] })
        if (-not $pending.Count) { break }
        $busy = @($pending | Where-Object { @(Worker $_).Count -gt 0 })
        foreach ($net in $pending) {
            if ($busy.Count -ge $Parallel) { break }
            if ($busy -contains $net) { continue }
            if ($attempts[$net] -gt $Retries) {
                $failed[$net] = $true
                Say "$net : gave up after $($attempts[$net]) attempts; see $queueLog and logs\${Prefix}_${net}_seed*.log"
                continue
            }
            $attempts[$net]++
            $sb = $ServerBase + 10 * [array]::IndexOf($Nets, $net)          # own TraCI ports for the non-RL runs
            $cmd = "& {0} -WorkerNet {1} -Seeds {2} -Episodes {3} -Prefix {4} -SimSeconds {5} -Python {6} -ServerBase {7}" -f
                (Q $PSCommandPath), (Q $net), ($Seeds -join ','), $Episodes, (Q $Prefix), $SimSeconds, (Q $Python), $sb
            if ($Extra.Count) { $cmd += " -Extra " + (($Extra | ForEach-Object { Q $_ }) -join ',') }
            if ($NonRLPython) { $cmd += " -NonRLPython " + (Q $NonRLPython) }
            if ($SkipNonRL) { $cmd += " -SkipNonRL" }
            Say ("{0} : start{1}" -f $net, $(if ($attempts[$net] -gt 1) { " (attempt $($attempts[$net]))" } else { '' }))
            Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd) | Out-Null
            $busy += $net
            Start-Sleep -Seconds 10
        }
        Start-Sleep -Seconds 60
    }
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $Python @h2hArgs --all 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $qdir "h2h_all.log")
    $ErrorActionPreference = $eap
    $bad = @($Nets | Where-Object { $failed[$_] })
    Say ("queue finished{0}; overview: results\h2h\overview.md" -f $(if ($bad.Count) { " (FAILED: $($bad -join ', '))" } else { '' }))
}
finally {
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
