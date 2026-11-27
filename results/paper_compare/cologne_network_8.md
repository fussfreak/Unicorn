# ARLGOS-v2 vs the Unicorn paper: Cologne (`cologne_network_8`)

**Queue length (primary): ARLGOS-v2 0.181 vs the paper's Unicorn-single 0.14: -29.6 %. Interval test: worse. Within 10 % (the G4 "match" rule): no.**

Positive % = ARLGOS-v2 better. ARLGOS-v2: 1 training seed(s), each run's best-on-validation checkpoint, greedy, on test seeds 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900. Paper: Table III, arXiv 2503.11488 v2 (HTML), Tables II-IV; v1 identical apart from CoLight rows.

## ARLGOS-v2 vs Unicorn-single

| Metric | ARLGOS-v2 | 95 % interval (mean over test seeds) | Paper Unicorn-single | ARLGOS-v2 better by | Verdict | Harness check |
|---|---|---|---|---|---|---|
| Queue length (veh) (lower is better) | 0.181 | [0.175, 0.188] | 0.14 | -29.6 % | worse | ok |
| Speed (m/s) (higher is better) | 8.364 | [8.316, 8.411] | 8.35 | +0.2 % | within noise | ok |
| Intersection delay (s) (lower is better) | 1.515 | [1.340, 1.690] | 0.67 | -126.1 % | worse | ok |
| Trip completion (veh/s) (higher is better) | 0.559 | [0.558, 0.559] | 0.56 | -0.2 % | worse | ok |
| Trip time (s) (lower is better) | 91.505 | [90.887, 92.122] | 89.98 | -1.7 % | worse | ok |
| Trip delay (s) (lower is better) | 26.003 | [25.499, 26.505] | 8.25 | -215.2 % | worse | ok |

## Is this harness the paper's harness? (check added after the first results)

Trip time minus trip delay is how long the average finished trip would take with no delay at all. No controller can change it, so within one simulator setup it should be about the same in every row. A clear gap between our rows and the paper's means the two setups differ (SUMO version, vehicle or route settings), and then speed, trip time and trip delay cannot be compared with the PDF.

| Row | Trip time - trip delay (s) |
|---|---|
| ARLGOS-v2 (ours) | 65.5 |
| Fixed-Time (ours) | 68.1 |
| Greedy (ours) | 65.5 |
| Max-Pressure (ours) | 66.4 |
| IDQN (paper) | 81.2 |
| IPPO (paper) | 80.8 |
| FMA2C (paper) | 103.8 |
| CoLight (paper) | 84.3 |
| MPLight (paper) | 83.1 |
| GESA-single (paper) | 81.8 |
| GESA-multiple (paper) | 81.1 |
| Unicorn-single (paper) | 81.7 |
| Unicorn-multiple (paper) | 82.8 |

"Harness check: suspect" in the first table = one of our own non-learning controllers already beats the paper's Unicorn on that metric, which points to a harness difference rather than a real gain.

## Every row in the paper's format: pooled mean (std)

| Row | Source | Queue length (veh) | Speed (m/s) | Intersection delay (s) | Trip completion (veh/s) | Trip time (s) | Trip delay (s) |
|---|---|---|---|---|---|---|---|
| **ARLGOS-v2 (1 seeds pooled)** | ours, this harness | **0.18** (0.15) | **8.36** (1.04) | **1.52** (1.76) | **0.56** (0.75) | **91.50** (56.13) | **26.00** (23.56) |
| ARLGOS-v2 seed1 (ep 1149) | ours, this harness | 0.18 (0.15) | 8.36 (1.04) | 1.52 (1.76) | 0.56 (0.75) | 91.50 (56.13) | 26.00 (23.56) |
| Fixed-Time | ours, same harness and seeds | 1.34 (0.69) | 4.26 (1.52) | 7.90 (3.71) | 0.55 (0.75) | 199.84 (175.03) | 131.72 (151.25) |
| Greedy | ours, same harness and seeds | 0.20 (0.15) | 8.14 (1.01) | 0.90 (0.73) | 0.56 (0.74) | 94.01 (55.14) | 28.55 (22.06) |
| Max-Pressure | ours, same harness and seeds | 0.43 (0.25) | 6.96 (1.13) | 3.90 (3.42) | 0.56 (0.75) | 112.59 (67.76) | 46.19 (37.63) |
| IDQN | paper, Table III | 0.20 (0.12) | 8.11 (0.70) | 3.18 (3.47) | 0.56 (0.35) | 92.99 (75.53) | 11.76 (53.13) |
| IPPO | paper, Table III | 0.15 (0.10) | 8.35 (0.67) | 0.69 (0.54) | 0.56 (0.35) | 89.17 (52.64) | 8.38 (10.91) |
| FMA2C | paper, Table III | 1.34 (0.60) | 4.50 (0.76) | 17.86 (6.58) | 0.55 (0.29) | 186.57 (166.87) | 82.72 (120.48) |
| CoLight | paper, Table III | 0.22 (0.13) | 7.74 (0.76) | 0.88 (0.49) | 0.56 (0.36) | 96.65 (54.96) | 12.39 (13.18) |
| MPLight | paper, Table III | 0.22 (0.13) | 7.81 (0.69) | 0.95 (0.60) | 0.56 (0.34) | 95.43 (54.82) | 12.37 (13.03) |
| GESA-single | paper, Table III | 0.14 (0.11) | 8.40 (0.87) | 2.38 (3.18) | 0.56 (0.35) | 89.52 (54.10) | 7.71 (21.54) |
| GESA-multiple | paper, Table III | 0.12 (0.11) | 8.52 (0.79) | 0.52 (0.42) | 0.56 (0.35) | 87.96 (49.85) | 6.82 (9.88) |
| **Unicorn-single** | paper, Table III | **0.14** (0.11) | **8.35** (0.73) | **0.67** (0.55) | **0.56** (0.35) | **89.98** (52.22) | **8.25** (11.55) |
| Unicorn-multiple | paper, Table III | 0.16 (0.13) | 8.22 (0.78) | 0.72 (0.51) | 0.56 (0.35) | 91.94 (53.39) | 9.18 (12.62) |

## Rule and limits (fixed before the test results)

* "better" / "worse": the paper's mean lies outside ARLGOS-v2's 95 % interval over the 10 test seeds (each averaged over training seeds); otherwise "within noise". "Match" (queue) = within 10 %.
* The PDF has no per-seed numbers, so there is no p-value against it, and its values are rounded to 2 decimals (a queue of 0.37 can be 0.365-0.375).
* The paper does not state its SUMO version, its number of training runs, or whether test actions are sampled; ARLGOS-v2 is tested greedily on SUMO 1.28. On Cologne8 SUMO 1.16 vs 1.28 moved every metric by less than 5 % (T3).
* The paper's rows are not rerun here; the supervisor's Unicorn rerun replaces them when it exists.

Runs: arlgos_cologne_network_8_seed1 (checkpoint ep 1149, val 0.1899)
