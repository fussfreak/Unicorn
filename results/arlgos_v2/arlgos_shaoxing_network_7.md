# ARLGOS-v2 on shaoxing_network_7 (3 training seeds)

Each cell: mean over the 10 test seeds of that run's chosen checkpoint. Last column: mean over the
training seeds. Baseline rows and the Unicorn rerun are added by the head-to-head report (T15/T17).

| Metric | seed1 | seed2 | seed3 | Mean |
|---|---|---|---|---|
| Queue length (veh) | 0.376 | 0.380 | 0.375 | 0.377 |
| Speed (m/s) | 4.119 | 4.139 | 4.108 | 4.122 |
| Intersection delay (s) | 13.872 | 14.345 | 14.765 | 14.328 |
| Trip completion (veh/s) | 1.209 | 1.209 | 1.205 | 1.208 |
| Trip time (s) | 228.883 | 228.220 | 226.462 | 227.855 |
| Trip delay (s) | 130.444 | 129.855 | 128.390 | 129.563 |

Chosen checkpoints: seed1 ep 899 (val 0.3855), seed2 ep 849 (val 0.3891), seed3 ep 1099 (val 0.3842)
