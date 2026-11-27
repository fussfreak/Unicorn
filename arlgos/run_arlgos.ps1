# =============================================================================
#  run_arlgos.ps1  --  ARLGOS-v2 on one of Unicorn's benchmark networks (T14/T15)
#
#  Trains -Seeds side by side (one process and one CPU thread each), then tests
#  each run's best-on-validation checkpoint once on Unicorn's 10 test seeds and
#  writes one summary table.
#
#    train   driver_arlgos.py      -> Train_MATSC\<Prefix>_<Net>_seed<k>\
#    test    arlgos\eval_arlgos.py -> ...\eval\test.md
#    table   results\arlgos_v2\<Prefix>_<Net>.md
#
#  Runs unattended:
#    * Windows is kept awake while it runs.
#    * Each training runs in its own hidden process and logs to logs\<name>.log.
#      Closing this window, or the progress window, does not stop a training.
#    * A progress window opens and shows every run's latest line (-NoWindow: none).
#    * Safe to re-run after a crash, a reboot or a closed window: finished runs are
#      skipped, stopped runs resume from their last saved state (at most
#      5 episodes lost), runs already going are left alone, and missing tests are run.
#    * A training that crashes is restarted from its last saved state, up to
#      -Retries times.
#
#  Usage (from anywhere):
#    .\arlgos\run_arlgos.ps1 -Net cologne_network_8                      # 3 seeds, 1500 episodes
#    .\arlgos\run_arlgos.ps1 -Net arterial_network_4_4 -Seeds 1          # one seed first
#    .\arlgos\run_arlgos.ps1 -Net cologne_network_8 -Prefix arlgos_selfnb -Extra "--conv","self_nb"
#    .\arlgos\run_arlgos.ps1 -Net cologne_network_8 -Episodes 3 -SimSeconds 300 -Prefix smoke   # pipeline check
#    .\arlgos\run_arlgos.ps1 -Net cologne_network_8 -Stop                # end this network's runs
#  -Python: an env with libsumo (unicorn_libsumo) is about 2.5x faster than traci (unicorn).
# =============================================================================
param(
    [string]$Net = "cologne_network_8",
    [int[]]$Seeds = @(1, 2, 3),
    [int]$Episodes = 1500,
    [string]$Prefix = "arlgos",
    [string[]]$Extra = @(),
    [int]$SimSeconds = 0,
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn_libsumo\python.exe",
    [int]$Retries = 3,
    [switch]$NoTest,
    [switch]$NoWindow,
    [switch]$Stop,
    [ValidateSet('published', 'heldout')][string]$Demand = 'published'
)
$ErrorActionPreference = 'Stop'
# T15F: -Demand heldout trains and tests on held-out traffic days (Fenglin only; see
# maps\shaoxing_network_7\build_file.py); those runs get their own prefix. Fenglin's trajectory
# file fcd.xml stays off (UNICORN_FCD=0; arlgos\fenglin_check.py shows nothing else changes).
if ($Demand -eq 'heldout' -and $Prefix -eq 'arlgos') { $Prefix = 'arlgos_heldout' }
$env:UNICORN_DEMAND = $Demand
$env:UNICORN_FCD = '0'
$fork = Split-Path $PSScriptRoot -Parent
Set-Location $fork
New-Item -ItemType Directory -Force "logs" | Out-Null
$queueLog = "logs\{0}_{1}.queue.log" -f $Prefix, $Net
function Say($m) {
    # a reader holding the file (e.g. Git Bash's tail -F) makes Add-Content fail; under 'Stop' that
    # killed the launcher (2026-11-22), so a log write never stops a run
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m
    Write-Host $line
    try { Add-Content $queueLog $line } catch { Write-Host "  (could not write $queueLog)" }
}
function Name($s)   { "{0}_{1}_seed{2}" -f $Prefix, $Net, $s }
function RunDir($s) { Join-Path "Train_MATSC" (Name $s) }
function Done($s)   { Test-Path (Join-Path (RunDir $s) "done.json") }
function Tested($s) { Test-Path (Join-Path (RunDir $s) "eval\test.json") }
function Running($s) {
    # a driver_arlgos.py process whose command line names this run
    $n = Name $s
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
      Where-Object { $_.CommandLine -match 'driver_arlgos\.py' -and $_.CommandLine -match ("--name\s+`"?" + [regex]::Escape($n) + "(`"|\s|$)") })
}

if ($Stop) {
    foreach ($s in $Seeds) {
        foreach ($p in (Running $s)) { Stop-Process -Id $p.ProcessId -Force; Say "stopped $(Name $s) (pid $($p.ProcessId)); re-run to resume" }
    }
    return
}

$env:PYTHONPATH = ''              # the standalone SUMO installer's tools folder would shadow the env's packages
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:SUMO_HOME = (& $Python -c "import sumo; print(sumo.SUMO_HOME)").Trim()
$hasLibsumo = (& $Python -c "import importlib.util as u; print(int(u.find_spec('libsumo') is not None))").Trim()
$env:UNICORN_LIBSUMO = $(if ($hasLibsumo -eq '1') { '1' } else { '' })

function Start-Training($s) {
    $n = Name $s
    $a = @("-u", "driver_arlgos.py", "--net", $Net, "--seed", $s, "--episodes", $Episodes, "--name", $n) + $Extra
    if ($SimSeconds -gt 0) { $a += @("--sim_seconds", $SimSeconds) }
    $latest = Join-Path (RunDir $s) "model\latest.pt"
    if (Test-Path $latest) { $a += "--resume" }
    elseif (Test-Path (RunDir $s)) {                                                 # stopped before its first save
        if (@(Running $s).Count -gt 0) { throw "$n is running; refusing to delete its folder" }
        Remove-Item -Recurse -Force (RunDir $s)
    }
    $log = "logs\$n.log"
    try { Add-Content $log ("[{0}] launcher: {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($a -join ' ')) }
    catch { Say "note: could not write to $log ($_)" }
    # cmd /c so the output can be appended to the log; hidden window = its own console,
    # which is what keeps the training alive when this window is closed
    # (cmd /c strips the outermost pair of quotes, hence the extra pair around the whole command)
    $cmd = '/c ""{0}" {1} >> "{2}" 2>&1"' -f $Python, ($a -join ' '), $log
    Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $fork -WindowStyle Hidden -PassThru
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
try {
    Say ("{0} on {1}: seeds {2}, {3} episodes, client {4}, extra '{5}'" -f $Prefix, $Net, ($Seeds -join ','), $Episodes,
         $(if ($env:UNICORN_LIBSUMO) { 'libsumo' } else { 'traci' }), ($Extra -join ' '))
    if (-not $NoWindow) {
        # progress window: the last line of each run's log, every 30 s; reading only
        $logs = ($Seeds | ForEach-Object { "'logs\$(Name $_).log'" }) -join ','
        $watch = "Set-Location '$fork'; `$host.UI.RawUI.WindowTitle = 'ARLGOS-v2 $Net (closing this window stops nothing)'; " +
                 "while (`$true) { Clear-Host; Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; foreach (`$l in @($logs)) { " +
                 "if (Test-Path `$l) { Write-Host ''; Write-Host `$l -ForegroundColor Cyan; Get-Content `$l -Tail 4 } }; Start-Sleep 30 }"
        Start-Process powershell -ArgumentList "-NoProfile", "-Command", $watch | Out-Null
    }
    $attempts = @{}
    foreach ($s in $Seeds) { $attempts[$s] = 0 }
    while ($true) {
        $waiting = $false
        foreach ($s in $Seeds) {
            if (Done $s) { continue }
            $waiting = $true
            # @(...) at the call site: Windows PowerShell 5.1 unrolls a one-element array returned
            # from a function, and a single CimInstance has no .Count there ($null -gt 0 is false).
            # Without it a running seed looked stopped and was restarted (T15F, 2026-11-16).
            if (@(Running $s).Count -gt 0) { continue }
            if ($attempts[$s] -gt $Retries) { throw "$(Name $s) stopped $($attempts[$s]) times; see logs\$(Name $s).log" }
            Say ("{0} {1}" -f $(if ($attempts[$s]) { "restart ($($attempts[$s]) of $Retries)" } else { "start" }), (Name $s))
            $attempts[$s]++
            Start-Training $s | Out-Null
            Start-Sleep -Seconds 5
        }
        if (-not $waiting) { break }
        Start-Sleep -Seconds 60
    }
    Say "training finished"
    if (-not $NoTest) {
        foreach ($s in $Seeds) {
            if (Tested $s) { continue }
            Say "test $(Name $s)"
            $t = @("-u", "arlgos\eval_arlgos.py", "--run", (RunDir $s))
            if ($SimSeconds -gt 0) { $t += @("--sim_seconds", $SimSeconds, "--n_seeds", 2) }     # pipeline check
            # Windows PowerShell 5.1 (the detached launch) turns any stderr line of a native command into a
            # terminating error under 'Stop' when it is redirected with 2>&1 (checked 2026-11-22); the exit
            # code below is what decides failure
            $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            & $Python @t 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append -FilePath ("logs\{0}.test.log" -f (Name $s)) | Select-String "TEST|Error"
            $code = $LASTEXITCODE
            $ErrorActionPreference = $eap
            if ($code -ne 0) { throw "eval_arlgos.py failed for $(Name $s); see logs\$(Name $s).test.log" }
        }
        & $Python -u arlgos\eval_arlgos.py --summary $Net --prefix $Prefix
        Say ("done: results\arlgos_v2\{0}_{1}.md" -f $Prefix, $Net)
        if ($Net -eq 'shaoxing_network_7' -and $SimSeconds -le 0) {       # T15F head-to-head table
            & $Python arlgos\fenglin_report.py --demand $Demand | Out-Null
            Say ("table: results\T15F\fenglin_h2h{0}.md" -f $(if ($Demand -eq 'heldout') { '_heldout' } else { '' }))
        }
        if ($SimSeconds -le 0 -and $Demand -eq 'published') {           # 2026-11-22: vs the paper's published rows
            $tag = $(if ($Prefix -eq 'arlgos') { '' } else { "_$Prefix" })
            $pc = @('arlgos\paper_compare.py', '--net', $Net, '--prefix', $Prefix, '--quiet', '--skip_unknown')
            if ($tag) { $pc += @('--tag', $tag) }                      # (Windows PowerShell drops empty arguments)
            & $Python @pc
            if (Test-Path ("results\paper_compare\{0}{1}.md" -f $Net, $tag)) {
                Say ("paper comparison: results\paper_compare\{0}{1}.md" -f $Net, $tag)
            }
        }
    }
}
finally {
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
