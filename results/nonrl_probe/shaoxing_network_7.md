# Non-RL headroom probe

Unicorn's own non-RL controllers (`evaluator_non_rl.py` via `arlgos/version_check.py run`,
unchanged) on its 10 test seeds (1000, 1100, ..., 1900), 3600 s each, SUMO 1.28.0.

## shaoxing_network_7

Mean over the 10 seeds [95 % CI], then the pooled mean (std) over all steps or trips, as in
the paper's tables.

| Controller | Queue length (veh) | Speed (m/s) | Intersection delay (s) | Trip completion (veh/s) | Trip time (s) | Trip delay (s) |
|---|---|---|---|---|---|---|
| Fixed-Time | 0.73 [0.73, 0.74]<br>0.73 (0.22) | 3.75 [3.73, 3.78]<br>3.75 (1.46) | 14.10 [13.49, 14.71]<br>14.10 (6.60) | 1.19 [1.19, 1.19]<br>1.19 (1.15) | 264.52 [262.95, 266.10]<br>264.52 (391.87) | 167.02 [165.48, 168.56]<br>167.02 (343.68) |
| Greedy | 0.51 [0.49, 0.52]<br>0.51 (0.14) | 4.01 [3.93, 4.08]<br>4.01 (1.59) | 32.67 [25.19, 40.14]<br>32.67 (33.33) | 1.19 [1.18, 1.21]<br>1.19 (1.12) | 231.32 [224.21, 238.43]<br>231.42 (387.07) | 133.15 [126.27, 140.02]<br>133.24 (343.26) |
| Max-Pressure | 0.51 [0.49, 0.52]<br>0.51 (0.14) | 4.01 [3.93, 4.08]<br>4.01 (1.59) | 32.67 [25.20, 40.13]<br>32.67 (33.34) | 1.19 [1.18, 1.21]<br>1.19 (1.12) | 231.06 [223.79, 238.34]<br>231.16 (386.69) | 132.91 [125.91, 139.91]<br>133.00 (342.84) |

**Reference only (from the PDF, not rerun):** the paper's learned rows for this network, pooled mean (std).

| Paper row | Queue length (veh) | Speed (m/s) | Intersection delay (s) | Trip completion (veh/s) | Trip time (s) | Trip delay (s) |
|---|---|---|---|---|---|---|
| GESA-single | 0.77 (0.28) | 3.76 (1.60) | 57.88 (21.72) | 1.23 (0.66) | 255.55 (391.00) | 118.14 (289.99) |
| GESA-multiple | 0.85 (0.34) | 3.91 (1.58) | 57.91 (22.95) | 1.24 (0.64) | 249.42 (381.99) | 112.36 (281.71) |
| Unicorn-single | 0.37 (0.08) | 3.54 (1.54) | 20.58 (6.32) | 1.20 (0.64) | 233.45 (417.07) | 104.55 (322.79) |
| Unicorn-multiple | 0.37 (0.09) | 3.56 (1.55) | 20.74 (6.98) | 1.20 (0.63) | 232.14 (408.46) | 102.72 (312.12) |

## Runs

| Network | Controller | SUMO reported | Green/yellow/teleport | s per episode |
|---|---|---|---|---|
| shaoxing_network_7 | FIXED | SUMO 1.28.0 | 15/5/600 | 474 |
| shaoxing_network_7 | GREEDY | SUMO 1.28.0 | 15/5/600 | 455 |
| shaoxing_network_7 | PRESSURE | SUMO 1.28.0 | 15/5/600 | 455 |
