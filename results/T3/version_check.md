# SUMO version check (T3)

Unicorn's own non-RL controllers (`evaluator_non_rl.py`, unchanged) on its 10 test seeds
(1000, 1100, ..., 1900), 3600 s each, with SUMO 1.16.0 and 1.28.0. Each run recorded the version SUMO
itself reported over TraCI. No training is involved, so any difference comes from the simulator.

## 1. Simulator effect: SUMO 1.28.0 minus 1.16.0, paired by seed

Mean of the per-seed differences, with a 95 % CI (t, 9 df) and a two-sided Wilcoxon
signed-rank p (descriptive, not corrected for the many cells). Relative = difference / 1.16.0 mean.

### cologne_network_8

| Agent | Metric | SUMO 1.16.0 | SUMO 1.28.0 | Difference [95 % CI] | Relative | p |
|---|---|---|---|---|---|---|
| Fixed-Time | Queue length (veh) | 1.305 | 1.339 | +0.034 [+0.004, +0.064] | +2.6 % | 0.037 |
| Fixed-Time | Speed (m/s) | 4.256 | 4.264 | +0.008 [-0.065, +0.080] | +0.2 % | 0.770 |
| Fixed-Time | Intersection delay (s) | 7.544 | 7.896 | +0.352 [+0.169, +0.536] | +4.7 % | 0.006 |
| Fixed-Time | Trip completion (veh/s) | 0.548 | 0.546 | -0.002 [-0.003, -0.000] | -0.3 % | 0.036 |
| Fixed-Time | Trip time (s) | 195.821 | 199.846 | +4.024 [+0.075, +7.974] | +2.1 % | 0.049 |
| Fixed-Time | Trip delay (s) | 128.032 | 131.728 | +3.695 [-0.035, +7.426] | +2.9 % | 0.064 |
| Greedy | Queue length (veh) | 0.207 | 0.203 | -0.004 [-0.013, +0.004] | -2.0 % | 0.322 |
| Greedy | Speed (m/s) | 7.860 | 8.145 | +0.285 [+0.239, +0.331] | +3.6 % | 0.002 |
| Greedy | Intersection delay (s) | 0.873 | 0.904 | +0.031 [-0.017, +0.078] | +3.5 % | 0.232 |
| Greedy | Trip completion (veh/s) | 0.560 | 0.560 | +0.000 [-0.000, +0.000] | +0.0 % | 0.490 |
| Greedy | Trip time (s) | 95.656 | 94.009 | -1.647 [-2.482, -0.812] | -1.7 % | 0.004 |
| Greedy | Trip delay (s) | 30.169 | 28.548 | -1.621 [-2.420, -0.822] | -5.4 % | 0.004 |
| Max-Pressure | Queue length (veh) | 0.427 | 0.428 | +0.001 [-0.009, +0.012] | +0.3 % | 0.846 |
| Max-Pressure | Speed (m/s) | 6.791 | 6.962 | +0.171 [+0.142, +0.200] | +2.5 % | 0.002 |
| Max-Pressure | Intersection delay (s) | 3.756 | 3.903 | +0.147 [-0.211, +0.504] | +3.9 % | 0.375 |
| Max-Pressure | Trip completion (veh/s) | 0.556 | 0.557 | +0.001 [-0.000, +0.002] | +0.1 % | 0.154 |
| Max-Pressure | Trip time (s) | 113.574 | 112.593 | -0.981 [-1.717, -0.245] | -0.9 % | 0.020 |
| Max-Pressure | Trip delay (s) | 47.182 | 46.191 | -0.991 [-1.679, -0.303] | -2.1 % | 0.014 |

## Runs

| SUMO asked | Network | Agent | SUMO reported | Green/yellow/teleport | s per episode |
|---|---|---|---|---|---|
| 1.16.0 | cologne_network_8 | FIXED | SUMO 1.16.0 | 15/5/-1 | 168 |
| 1.16.0 | cologne_network_8 | GREEDY | SUMO 1.16.0 | 15/5/-1 | 116 |
| 1.16.0 | cologne_network_8 | PRESSURE | SUMO 1.16.0 | 15/5/-1 | 131 |
| 1.28.0 | cologne_network_8 | FIXED | SUMO 1.28.0 | 15/5/-1 | 161 |
| 1.28.0 | cologne_network_8 | GREEDY | SUMO 1.28.0 | 15/5/-1 | 109 |
| 1.28.0 | cologne_network_8 | PRESSURE | SUMO 1.28.0 | 15/5/-1 | 124 |

## Grid 5x5 setup check: moved to T4 (lab PC)

The Grid 5x5 runs (the comparison with Table II) were started twice on the laptop on 2026-11-03.
The first attempt was stopped when the Claude session restarted. In the second, all six SUMO processes
died within 9 s of each other, most likely because the laptop ran out of RAM (0.7 GB free afterwards).
The check now runs on the lab PC in T4: `.\arlgos\version_check.ps1 -Nets grid_network_5_5`.
