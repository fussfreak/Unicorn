# =============================================================================
#  version_check.ps1  --  T3: how much does SUMO 1.16 vs 1.28 change the numbers?
#
#  Runs Unicorn's non-RL controllers (FIXED, GREEDY, PRESSURE; upstream
#  evaluator_non_rl.py's Evaluator) on its 10 test seeds under SUMO 1.16.0
#  (env unicorn_sumo116) and 1.28.0 (env unicorn): 6 processes in parallel per
#  network, Cologne8 first, then Grid 5x5 (the only network with non-RL rows in
#  the paper, Table II).  Then writes results\T3\version_check.{md,json}.
#
#  Raw data:  results\T3\raw\sumo-<version>\<net>\{eval_data\*.csv, <AGENT>.done.json}
#  Logs:      logs\vc_<version>_<net>_<AGENT>.log
#  Running it again skips finished runs (those with a .done.json), so Ctrl+C is safe.
#
#  Usage:   .\arlgos\version_check.ps1          # about 1-2 h on the laptop
# =============================================================================
param(
    [string[]]$Nets = @("cologne_network_8", "grid_network_5_5"),
    [string[]]$Agents = @("FIXED", "GREEDY", "PRESSURE"),
    [string]$EnvRoot = "C:\Users\DELL\miniconda3\envs"
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$versions = [ordered]@{ "1.16.0" = (Join-Path $EnvRoot "unicorn_sumo116\python.exe");
                        "1.28.0" = (Join-Path $EnvRoot "unicorn\python.exe") }
$raw = "results\T3\raw"
$env:PYTHONPATH = ''              # the standalone SUMO installer's tools folder would shadow the env's traci
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force logs | Out-Null
function Stamp { Get-Date -Format HH:mm }

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
$procs = @()
try {
    for ($n = 0; $n -lt $Nets.Count; $n++) {
        $net = $Nets[$n]
        $procs = @()
        $v = 0
        foreach ($version in $versions.Keys) {
            $python = $versions[$version]
            $env:SUMO_HOME = (& $python -c "import sumo; print(sumo.SUMO_HOME)").Trim()   # inherited by the child
            for ($a = 0; $a -lt $Agents.Count; $a++) {
                $agent = $Agents[$a]
                $out = Join-Path $raw "sumo-$version\$net"
                if (Test-Path (Join-Path $out "$agent.done.json")) { continue }
                $server = 70 + 10 * $v + 3 * $n + $a                         # TraCI port 6000+server
                $log = "logs\vc_${version}_${net}_${agent}.log"
                Write-Host ("[{0}] SUMO {1} {2} {3} (server {4})" -f (Stamp), $version, $net, $agent, $server)
                $procs += Start-Process -FilePath $python -NoNewWindow -PassThru -WorkingDirectory $PWD.Path `
                    -ArgumentList @("-u", "arlgos\version_check.py", "run", "--net", $net, "--agent", $agent,
                                    "--server", $server, "--out", $out) `
                    -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$', '.err.log')
            }
            $v++
        }
        $procs | Wait-Process
        foreach ($version in $versions.Keys) {
            foreach ($agent in $Agents) {
                if (-not (Test-Path (Join-Path $raw "sumo-$version\$net\$agent.done.json"))) {
                    throw "SUMO $version $net $agent did not finish: see logs\vc_${version}_${net}_${agent}.err.log"
                }
            }
        }
    }
    Write-Host ("[{0}] summary" -f (Stamp))
    $env:SUMO_HOME = (& $versions["1.28.0"] -c "import sumo; print(sumo.SUMO_HOME)").Trim()
    & $versions["1.28.0"] -u arlgos\version_check.py summarize --root $raw --out results\T3\version_check
    if ($LASTEXITCODE -ne 0) { throw "summarize exited $LASTEXITCODE" }
    Write-Host ("[{0}] Done: results\T3\version_check.md" -f (Stamp))
}
finally {
    foreach ($p in $procs) { if ($p -and -not $p.HasExited) { $p.Kill() } }
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
