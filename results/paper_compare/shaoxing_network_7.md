# ARLGOS-v2 vs the Unicorn paper: Fenglin (`shaoxing_network_7`)

**Queue length (primary): ARLGOS-v2 0.377 vs the paper's Unicorn-single 0.37: -1.8 %. Interval test: worse. Within 10 % (the G4 "match" rule): yes, a match.**

Positive % = ARLGOS-v2 better. ARLGOS-v2: 3 training seed(s), each run's best-on-validation checkpoint, greedy, on test seeds 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900. Paper: Table IV, arXiv 2503.11488 v2 (HTML), Tables II-IV; v1 identical apart from CoLight rows.

## ARLGOS-v2 vs Unicorn-single

| Metric | ARLGOS-v2 | 95 % interval (mean over test seeds) | Paper Unicorn-single | ARLGOS-v2 better by | Verdict | Harness check |
|---|---|---|---|---|---|---|
| Queue length (veh) (lower is better) | 0.377 | [0.374, 0.379] | 0.37 | -1.8 % | worse | ok |
| Speed (m/s) (higher is better) | 4.122 | [4.103, 4.141] | 3.54 | +16.4 % | better | suspect: Fixed-Time, Greedy, Max-Pressure (ours) also beats the paper |
| Intersection delay (s) (lower is better) | 14.328 | [13.735, 14.920] | 20.58 | +30.4 % | better | suspect: Fixed-Time (ours) also beats the paper |
| Trip completion (veh/s) (higher is better) | 1.208 | [1.206, 1.209] | 1.20 | +0.6 % | better | ok |
| Trip time (s) (lower is better) | 227.856 | [226.774, 228.936] | 233.45 | +2.4 % | better | suspect: Greedy, Max-Pressure (ours) also beats the paper |
| Trip delay (s) (lower is better) | 129.563 | [128.462, 130.663] | 104.55 | -23.9 % | worse | ok |

## Is this harness the paper's harness? (check added after the first results)

Trip time minus trip delay is how long the average finished trip would take with no delay at all. No controller can change it, so within one simulator setup it should be about the same in every row. A clear gap between our rows and the paper's means the two setups differ (SUMO version, vehicle or route settings), and then speed, trip time and trip delay cannot be compared with the PDF.

| Row | Trip time - trip delay (s) |
|---|---|
| ARLGOS-v2 (ours) | 98.3 |
| Fixed-Time (ours) | 97.5 |
| Greedy (ours) | 98.2 |
| Max-Pressure (ours) | 98.2 |
| GESA-single (paper) | 137.4 |
| GESA-multiple (paper) | 137.1 |
| Unicorn-single (paper) | 128.9 |
| Unicorn-multiple (paper) | 129.4 |

"Harness check: suspect" in the first table = one of our own non-learning controllers already beats the paper's Unicorn on that metric, which points to a harness difference rather than a real gain.

## Every row in the paper's format: pooled mean (std)

| Row | Source | Queue length (veh) | Speed (m/s) | Intersection delay (s) | Trip completion (veh/s) | Trip time (s) | Trip delay (s) |
|---|---|---|---|---|---|---|---|
| **ARLGOS-v2 (3 seeds pooled)** | ours, this harness | **0.38** (0.10) | **4.12** (1.44) | **14.33** (7.46) | **1.21** (1.13) | **227.86** (386.94) | **129.56** (342.49) |
| ARLGOS-v2 seed1 (ep 899) | ours, this harness | 0.38 (0.10) | 4.12 (1.44) | 13.87 (6.70) | 1.21 (1.13) | 228.88 (387.54) | 130.44 (342.87) |
| ARLGOS-v2 seed2 (ep 849) | ours, this harness | 0.38 (0.10) | 4.14 (1.43) | 14.34 (7.61) | 1.21 (1.13) | 228.22 (386.87) | 129.86 (342.26) |
| ARLGOS-v2 seed3 (ep 1099) | ours, this harness | 0.37 (0.10) | 4.11 (1.44) | 14.77 (7.97) | 1.20 (1.13) | 226.46 (386.42) | 128.39 (342.35) |
| Fixed-Time | ours, same harness and seeds | 0.73 (0.22) | 3.75 (1.46) | 14.10 (6.60) | 1.19 (1.15) | 264.52 (391.87) | 167.02 (343.68) |
| Greedy | ours, same harness and seeds | 0.51 (0.14) | 4.01 (1.59) | 32.67 (33.33) | 1.19 (1.12) | 231.42 (387.07) | 133.24 (343.26) |
| Max-Pressure | ours, same harness and seeds | 0.51 (0.14) | 4.01 (1.59) | 32.67 (33.34) | 1.19 (1.12) | 231.16 (386.69) | 133.00 (342.84) |
| GESA-single | paper, Table IV | 0.77 (0.28) | 3.76 (1.60) | 57.88 (21.72) | 1.23 (0.66) | 255.55 (391.00) | 118.14 (289.99) |
| GESA-multiple | paper, Table IV | 0.85 (0.34) | 3.91 (1.58) | 57.91 (22.95) | 1.24 (0.64) | 249.42 (381.99) | 112.36 (281.71) |
| **Unicorn-single** | paper, Table IV | **0.37** (0.08) | **3.54** (1.54) | **20.58** (6.32) | **1.20** (0.64) | **233.45** (417.07) | **104.55** (322.79) |
| Unicorn-multiple | paper, Table IV | 0.37 (0.09) | 3.56 (1.55) | 20.74 (6.98) | 1.20 (0.63) | 232.14 (408.46) | 102.72 (312.12) |

## Rule and limits (fixed before the test results)

* "better" / "worse": the paper's mean lies outside ARLGOS-v2's 95 % interval over the 10 test seeds (each averaged over training seeds); otherwise "within noise". "Match" (queue) = within 10 %.
* The PDF has no per-seed numbers, so there is no p-value against it, and its values are rounded to 2 decimals (a queue of 0.37 can be 0.365-0.375).
* The paper does not state its SUMO version, its number of training runs, or whether test actions are sampled; ARLGOS-v2 is tested greedily on SUMO 1.28. On Cologne8 SUMO 1.16 vs 1.28 moved every metric by less than 5 % (T3).
* The paper's rows are not rerun here; the supervisor's Unicorn rerun replaces them when it exists.

Runs: arlgos_shaoxing_network_7_seed1 (checkpoint ep 899, val 0.3855), arlgos_shaoxing_network_7_seed2 (checkpoint ep 849, val 0.3891), arlgos_shaoxing_network_7_seed3 (checkpoint ep 1099, val 0.3842)
