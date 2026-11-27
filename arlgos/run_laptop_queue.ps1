# =============================================================================
#  run_laptop_queue.ps1  --  ARLGOS-v2 runs on the laptop, ONE process at a time
#
#  Added 2026-11-22 (user: "one process at a time", the laptop ran hot with two).
#  Works through -Plan in order. For each network: trains each seed alone
#  (run_arlgos.ps1 -Seeds <k> -NoTest), then tests every seed of that network
#  once and writes its table (run_arlgos.ps1 -Seeds <all>; it skips what is done).
#
#    * Safe to re-run: finished seeds are skipped, stopped seeds resume from their
#      last saved state (at most 5 episodes replayed), finished tests are skipped.
#    * One progress window shows the newest logs (closing it stops nothing).
#    * -Stop ends the queue, its launcher and its training/test process, in that
#      order (the launcher restarts a stopped seed if it is left running).
#
#  Usage:
#    .\arlgos\run_laptop_queue.ps1                                  # default plan below
#    .\arlgos\run_laptop_queue.ps1 -Plan 'cologne_network_8:1,2,3'
#    .\arlgos\run_laptop_queue.ps1 -Stop
#    .\arlgos\run_laptop_queue.ps1 -DryRun                          # print what it would do
# =============================================================================
param(
    [string[]]$Plan = @('shaoxing_network_7:1,2,3', 'cologne_network_8:1'),
    [string[]]$Extra = @('--val_every', '50'),
    [switch]$NoWindow,
    [switch]$Stop,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$fork = Split-Path $PSScriptRoot -Parent
Set-Location $fork
New-Item -ItemType Directory -Force "logs" | Out-Null
$queueLog = "logs\laptop_queue.log"
$launcher = Join-Path $PSScriptRoot "run_arlgos.ps1"
function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m
    Write-Host $line
    try { Add-Content $queueLog $line } catch { }
}

if ($Stop) {
    # queue first, then launchers, then python: a launcher left alive restarts its seed
    $queue = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
               Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'run_laptop_queue\.ps1' })
    foreach ($p in $queue) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    $launchers = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
                   Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'run_arlgos\.ps1' })
    foreach ($p in $launchers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    $py = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match 'driver_arlgos\.py|eval_arlgos\.py|fenglin_report\.py' })
    foreach ($p in $py) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Say ("stopped: {0} queue, {1} launcher, {2} python process(es); re-run to resume" -f $queue.Count, $launchers.Count, $py.Count)
    return
}

$items = foreach ($entry in $Plan) {
    $net, $seedText = $entry -split ':', 2
    [pscustomobject]@{ Net = $net.Trim(); Seeds = @($seedText -split ',' | ForEach-Object { [int]$_.Trim() }) }
}
function Done($net, $s) { Test-Path (Join-Path "Train_MATSC" ("arlgos_{0}_seed{1}\done.json" -f $net, $s)) }

if ($DryRun) {
    foreach ($it in $items) {
        foreach ($s in $it.Seeds) { "{0} seed {1}: {2}" -f $it.Net, $s, $(if (Done $it.Net $s) { 'done, skip' } else { 'train (alone)' }) }
        "{0}: test seeds {1} once, then the table" -f $it.Net, ($it.Seeds -join ',')
    }
    return
}

if (-not $NoWindow) {
    # progress: the last lines of the 3 most recently written ARLGOS-v2 logs, every 30 s; reading only
    $watch = "Set-Location '$fork'; `$host.UI.RawUI.WindowTitle = 'ARLGOS-v2 laptop queue (closing this window stops nothing)'; " +
             "while (`$true) { Clear-Host; Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; Write-Host ''; Get-Content 'logs\laptop_queue.log' -Tail 4 -ErrorAction SilentlyContinue; " +
             "Get-ChildItem 'logs\arlgos_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 3 | ForEach-Object { " +
             "Write-Host ''; Write-Host `$_.Name -ForegroundColor Cyan; Get-Content `$_.FullName -Tail 3 }; Start-Sleep 30 }"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $watch | Out-Null
}

Say ("queue start: {0}" -f ($Plan -join ' then '))
foreach ($it in $items) {
    foreach ($s in $it.Seeds) {
        if (Done $it.Net $s) { continue }
        Say ("train {0} seed {1} (alone)" -f $it.Net, $s)
        & $launcher -Net $it.Net -Seeds $s -Extra $Extra -NoTest -NoWindow
    }
    Say ("test {0} seeds {1}" -f $it.Net, ($it.Seeds -join ','))
    & $launcher -Net $it.Net -Seeds $it.Seeds -Extra $Extra -NoWindow
}
Say "queue finished"
