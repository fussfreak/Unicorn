# =============================================================================
#  run_unicorn.ps1  --  one Unicorn training run (driver_unicorn.py), logged
#
#  Sets the per-run values that parameters.py reads (UNICORN_NET, _MAX_EPISODES,
#  _WORKERS, _SEED, _EXP_NAME), points SUMO_HOME at the env's own eclipse-sumo
#  (so the SUMO version is the env's, not C:\Program Files (x86)\SUMO), keeps
#  Windows awake and timestamps every output line (arlgos\run_logged.py).
#
#  One "episode" in Unicorn's counter = one PPO update = one 3600 s simulation
#  on each of the -Workers Ray workers.  -Episodes 4 gives 3 updates.
#
#  Output:  Train_MATSC\<Name>\{model,train,config.json}
#           logs\<Name>.log  (timestamped)   logs\<Name>.mem.csv  (RAM every 30 s)
#  Refuses to start if Train_MATSC\<Name> exists: runs are never overwritten.
#  -Resume continues that run instead, from model\checkpoint.pkl, which the
#  driver rewrites after every update (utils.save_latest, T4).  Give it the
#  same -Seed / -Net / -Workers / -Episodes as the original run.  At most the
#  update that was running is lost.  The workers then draw new training seeds,
#  so a resumed run is valid but not bit-identical to an uninterrupted one.
#
#  Usage (from anywhere; the unicorn env must exist):
#    .\arlgos\run_unicorn.ps1 -Episodes 4 -Name smoke_cologne8      # T3 smoke run
#    .\arlgos\run_unicorn.ps1 -Seed 1                                # full run, 1500 updates
#    .\arlgos\run_unicorn.ps1 -Seed 1 -Resume                        # after a reboot
# =============================================================================
param(
    [int]$Episodes = 1500,
    [int]$Seed = 1,
    [int]$Workers = 6,
    [string]$Net = "cologne_network_8",
    [string]$Name = "",
    [string]$Python = "C:\Users\DELL\miniconda3\envs\unicorn\python.exe",
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Name) { $Name = "{0}_seed{1}" -f $Net, $Seed }
$run = Join-Path "Train_MATSC" $Name
if ($Resume) {
    if (-not (Test-Path (Join-Path $run "model\checkpoint.pkl"))) {
        throw "$run\model\checkpoint.pkl not found: nothing to resume (runs started before T4 never wrote it)"
    }
    $env:UNICORN_RESUME = '1'
} else {
    if (Test-Path $run) { throw "$run already exists; pick another -Name (runs are never overwritten), or -Resume" }
    $env:UNICORN_RESUME = ''
}

$env:PYTHONPATH = ''              # the standalone SUMO installer's tools folder would shadow the env's traci
$env:PYTHONIOENCODING = 'utf-8'
$env:SUMO_HOME = (& $Python -c "import sumo; print(sumo.SUMO_HOME)").Trim()
$env:UNICORN_NET = $Net
$env:UNICORN_MAX_EPISODES = "$Episodes"
$env:UNICORN_WORKERS = "$Workers"
$env:UNICORN_SEED = "$Seed"
$env:UNICORN_EXP_NAME = $Name
$log = Join-Path "logs" "$Name.log"
Write-Host ("[{0}] {1}: {2} updates x {3} workers, seed {4}, SUMO_HOME={5}" -f
            (Get-Date -Format HH:mm), $Name, ($Episodes - 1), $Workers, $Seed, $env:SUMO_HOME)
Write-Host "log: $log"

# Windows sleeps an idle laptop after its timeout however busy the CPU is
# (run_A1.ps1 in ARLGOS).  Closing the lid can still put it to sleep.
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(2147483649) | Out-Null   # CONTINUOUS | SYSTEM_REQUIRED
try {
    & $Python -u arlgos\run_logged.py $log -- $Python -u driver_unicorn.py
    if ($LASTEXITCODE -ne 0) { throw "driver_unicorn.py exited $LASTEXITCODE; see $log" }
    Write-Host ("[{0}] Done: {1}" -f (Get-Date -Format HH:mm), $run)
}
finally {
    [Win32.Power]::SetThreadExecutionState(2147483648) | Out-Null
}
