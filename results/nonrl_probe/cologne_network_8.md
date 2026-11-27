# Non-RL headroom probe

Unicorn's own non-RL controllers (`evaluator_non_rl.py` via `arlgos/version_check.py run`,
unchanged) on its 10 test seeds (1000, 1100, ..., 1900), 3600 s each, SUMO 1.28.0.

## cologne_network_8

Mean over the 10 seeds [95 % CI], then the pooled mean (std) over all steps or trips, as in
the paper's tables.

| Controller | Queue length (veh) | Speed (m/s) | Intersection delay (s) | Trip completion (veh/s) | Trip time (s) | Trip delay (s) |
|---|---|---|---|---|---|---|
| Fixed-Time | 1.34 [1.31, 1.37]<br>1.34 (0.69) | 4.26 [4.20, 4.32]<br>4.26 (1.52) | 7.90 [7.69, 8.10]<br>7.90 (3.71) | 0.55 [0.55, 0.55]<br>0.55 (0.75) | 199.85 [196.86, 202.83]<br>199.84 (175.03) | 131.73 [128.87, 134.59]<br>131.72 (151.25) |
| Greedy | 0.20 [0.20, 0.21]<br>0.20 (0.15) | 8.14 [8.11, 8.17]<br>8.14 (1.01) | 0.90 [0.88, 0.93]<br>0.90 (0.73) | 0.56 [0.56, 0.56]<br>0.56 (0.74) | 94.01 [93.64, 94.38]<br>94.01 (55.14) | 28.55 [28.15, 28.95]<br>28.55 (22.06) |
| Max-Pressure | 0.43 [0.42, 0.44]<br>0.43 (0.25) | 6.96 [6.93, 6.99]<br>6.96 (1.13) | 3.90 [3.58, 4.22]<br>3.90 (3.42) | 0.56 [0.56, 0.56]<br>0.56 (0.75) | 112.59 [111.91, 113.28]<br>112.59 (67.76) | 46.19 [45.54, 46.84]<br>46.19 (37.63) |

## Runs

| Network | Controller | SUMO reported | Green/yellow/teleport | s per episode |
|---|---|---|---|---|
| cologne_network_8 | FIXED | SUMO 1.28.0 | 15/5/-1 | 161 |
| cologne_network_8 | GREEDY | SUMO 1.28.0 | 15/5/-1 | 109 |
| cologne_network_8 | PRESSURE | SUMO 1.28.0 | 15/5/-1 | 124 |
