# =============================================================================
#  nonrl_probe.ps1  --  how much headroom does a network leave above non-RL control?
#
#  Runs Unicorn's non-RL controllers (FIXED, GREEDY, PRESSURE; evaluator_non_rl.py's
#  Evaluator through arlgos\version_check.py run, unchanged) on its 10 test seeds,
#  SUMO 1.28.0 only (env unicorn), one process per controller, below-normal
#  priority.  Then arlgos\nonrl_probe.py summarize writes
#  results\nonrl_probe\<net>.{md,json} with the paper's learned rows beside them as
#  reference only.
#
#  First use (2026-11-05): shaoxing_network_7 = GESA's "Fenglin", a real 7-junction
#  corridor (each junction's only signalised neighbours are west and east).  The
#  paper's Table IV compares Unicorn there with GESA only, with no non-RL row.
#
#  Raw data:  results\nonrl_probe\raw\sumo-1.28.0\<net>\{eval_data\*.csv, <AGENT>.done.json}
#  Logs:      logs\probe_<net>_<AGENT>.log
#  Running it again skips finished controllers (those with a .done.json).
#
#  Usage:   .\arlgos\nonrl_probe.ps1                          # Fenglin, about 1 h
#           .\arlgos\nonrl_probe.ps1 -Net arterial_network_4_4
# =============================================================================
param(
    [string]$Net = "shaoxing_network_7",
    [string[]]$Agents = @("FIXED", "GREEDY", "PRESSURE"),
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn\python.exe",
    [int]$ServerBase = 90
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = ''              # the standalone SUMO installer's tools folder would shadow the env's traci
$env:PYTHONIOENCODING = 'utf-8'
$env:SUMO_HOME = (& $Python -c "import sumo; print(sumo.SUMO_HOME)").Trim()   # inherited by the children
$raw = "results\nonrl_probe\raw"
$out = Join-Path $raw "sumo-1.28.0\$Net"
New-Item -ItemType Directory -Force logs, $out | Out-Null
function Stamp { Get-Date -Format HH:mm }

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
$procs = @()
try {
    for ($a = 0; $a -lt $Agents.Count; $a++) {
        $agent = $Agents[$a]
        if (Test-Path (Join-Path $out "$agent.done.json")) { continue }
        $log = "logs\probe_${Net}_${agent}.log"
        Write-Host ("[{0}] {1} {2} (server {3})" -f (Stamp), $Net, $agent, ($ServerBase + $a))
        $p = Start-Process -FilePath $Python -NoNewWindow -PassThru -WorkingDirectory $PWD.Path `
            -ArgumentList @("-u", "arlgos\version_check.py", "run", "--net", $Net, "--agent", $agent,
                            "--server", ($ServerBase + $a), "--out", $out) `
            -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$', '.err.log')
        $p.PriorityClass = 'BelowNormal'
        $procs += $p
    }
    $procs | Wait-Process
    foreach ($agent in $Agents) {
        if (-not (Test-Path (Join-Path $out "$agent.done.json"))) {
            throw "$Net $agent did not finish: see logs\probe_${Net}_${agent}.err.log"
        }
    }
    & $Python -u arlgos\nonrl_probe.py summarize --root $raw --out "results\nonrl_probe\$Net"
    if ($LASTEXITCODE -ne 0) { throw "summarize exited $LASTEXITCODE" }
    Write-Host ("[{0}] Done: results\nonrl_probe\{1}.md" -f (Stamp), $Net)
}
finally {
    foreach ($p in $procs) { if ($p -and -not $p.HasExited) { $p.Kill() } }
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
