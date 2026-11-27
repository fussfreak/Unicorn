# T15F: ARLGOS-v2 vs Unicorn on Fenglin (published protocol (fixed route file))

Unicorn's 10 test seeds (1000, 1100, ..., 1900), 3600 s each, SUMO 1.28.0. Each cell: mean over
the test seeds of the mean over a method's training seeds. Every row is scored by the same
function (`arlgos/version_check.episode_metrics`). Lower is better except speed and completion.

| Method | Runs | Queue (veh) | Speed (m/s) | Int. delay (s) | Completion (veh/s) | Trip time (s) | Trip delay (s) |
|---|---|---|---|---|---|---|---|
| **ARLGOS-v2** (greedy) | 3 | 0.377 | 4.122 | 14.328 | 1.208 | 227.855 | 129.563 |
| Fixed-time | 1 | 0.734 | 3.755 | 14.101 | 1.188 | 264.525 | 167.021 |
| Greedy | 1 | 0.507 | 4.005 | 32.668 | 1.194 | 231.322 | 133.146 |
| Max-pressure | 1 | 0.507 | 4.005 | 32.667 | 1.194 | 231.064 | 132.910 |
| *Unicorn-single, paper Table IV (PDF, reference only)* | - | *0.37* | *3.54* | *20.58* | *1.20* | *233.45* | *104.55* |

## Runs

| Run | Chosen checkpoint | Validation queue | Test queue (greedy) |
|---|---|---|---|
| arlgos_shaoxing_network_7_seed1 | 899 | 0.3855 | 0.376 |
| arlgos_shaoxing_network_7_seed2 | 849 | 0.3891 | 0.380 |
| arlgos_shaoxing_network_7_seed3 | 1099 | 0.3842 | 0.375 |

**No comparison yet:** no tested Unicorn run in results/T15F/runs/.
