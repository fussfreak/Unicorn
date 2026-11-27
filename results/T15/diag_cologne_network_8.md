# Per-junction diagnosis: arlgos_cologne_network_8_seed1 on `cologne_network_8`

Checkpoint `best_val.pt` (episode 1149), greedy, vs Unicorn's Greedy controller, on validation seeds 5042, 5043, 5044, 5045, 5046 (the test seeds are not used). Lane weights: `max`. Written by `arlgos/diag_junctions.py`.

## Checks

* [PASS] sumo_link_order: SUMO's controlled links = traffic_movement_list at all 8 junctions
* [PASS] every_controlled_lane_is_a_node
* [PASS] audit_matches_env_lane_weights
* [PASS] padded_phases_zero_and_masked
* [PASS] aliased_pairs_identical_in_model: 6 aliased pair(s): the model's phase summary e_(j,p) is identical within each pair
* [PASS] light_shown_equals_tracked_phase: 0 mismatches over all decisions of both controllers
* [PASS] no_padded_phase_chosen
* [PASS] replay_reproduces_logged_validation: replay 0.189941 vs logged 0.189941 (checkpoint episode 1149)

## Whole network

| | ARLGOS-v2 | Greedy | ARLGOS-v2 - Greedy |
|---|---|---|---|
| Queue at decision points (the validation score) | 0.1899 | 0.2099 | -0.0200 |
| Queue every second (the test metric) | 0.1791 | 0.2023 | -0.0231 |
| Waiting time every second ("intersection delay", s) | 1.918 | 0.912 | +1.006 |

## Per junction

Queue and waiting-time columns are each junction's share of the network value, so they add up to the row above. Aliased = phase pairs with identical lane weights (subset phase in brackets).

| Junction | Phases | Lanes | Aliased | Queue share A / G | A - G | Wait share A / G (s) | Max wait A / G (s) | Switches per h A / G |
|---|---|---|---|---|---|---|---|---|
| 252017285 | 2 | 4 | - | 0.0183 / 0.0368 | -0.0185 | 0.090 / 0.162 | 43 / 28 | 138 / 186 |
| 32319828 | 2 | 2 | 0=1 [1] | 0.0000 / 0.0000 | +0.0000 | 0.000 / 0.000 | 2 / 2 | 0 / 0 |
| 62426694 | 3 | 4 | - | 0.0138 / 0.0185 | -0.0047 | 0.099 / 0.083 | 103 / 20 | 169 / 149 |
| 280120513 | 3 | 4 | - | 0.0125 / 0.0082 | +0.0043 | 0.106 / 0.036 | 118 / 20 | 133 / 70 |
| 256201389 | 3 | 3 | 0=1 [1] | 0.0010 / 0.0000 | +0.0010 | 0.009 / 0.000 | 45 / 0 | 232 / 0 |
| 26110729 | 4 | 6 | 2=3 [3] | 0.0673 / 0.0826 | -0.0154 | 0.357 / 0.352 | 85 / 76 | 167 / 176 |
| 247379907 | 4 | 6 | 2=3 [3] | 0.0486 / 0.0363 | +0.0122 | 1.132 / 0.149 | 493 / 28 | 134 / 165 |
| cluster_10... | 4 | 4 | 0=1 [1], 2=3 [3] | 0.0177 / 0.0198 | -0.0022 | 0.085 / 0.087 | 75 / 71 | 144 / 89 |
| elsewhere | | | | | | 0.040 / 0.042 | 48 / 57 | |

"A - G" = the junction's part of ARLGOS-v2 minus Greedy on the per-second queue: negative = ARLGOS-v2 has fewer halted cars there. The column adds up to -0.0231.

## Per lane

"Seen" = the share of the lane the 50 m camera covers. "red_t at cap" = the share of decisions at which the lane had been red for at least 180 s, where the agent's red_t input stops growing. Waiting = vehicle-seconds per hour with a wait of 60 s or more.

| Junction | Lane | Length (m) | Seen | Halting A / G | Longest red A / G (s) | red_t at cap A / G | Max wait A / G (s) | Waiting >= 60 s A / G |
|---|---|---|---|---|---|---|---|---|
| 252017285 | `-8716807#0_0` | 100 | 50 % | 0.12 / 0.23 | 150 / 135 | 0 % / 0 % | 40 / 20 | 0 / 0 |
| 252017285 | `133081985#1_0` | 83 | 60 % | 0.14 / 0.08 | 90 / 30 | 0 % / 0 % | 43 / 18 | 0 / 0 |
| 252017285 | `-23283579#0_0` | 62 | 81 % | 0.22 / 0.82 | 150 / 135 | 0 % / 0 % | 37 / 20 | 0 / 0 |
| 252017285 | `-28675510#0_0` | 123 | 41 % | 0.13 / 0.09 | 90 / 30 | 0 % / 0 % | 34 / 28 | 0 / 0 |
| 32319828 | `-4936412_0` | 34 | 100 % | 0.00 / 0.00 | 0 / 0 | 0 % / 0 % | 0 / 0 | 0 / 0 |
| 32319828 | `-23686088#0_0` | 37 | 100 % | 0.00 / 0.00 | 0 / 0 | 0 % / 0 % | 2 / 2 | 0 / 0 |
| 62426694 | `297047308_0` | 29 | 100 % | 0.17 / 0.04 | 0 / 0 | 0 % / 0 % | 103 / 17 | 13 / 0 |
| 62426694 | `-28675494#1_0` | 73 | 68 % | 0.03 / 0.02 | 120 / 30 | 0 % / 0 % | 39 / 16 | 0 / 0 |
| 62426694 | `-28675494#1_1` | 73 | 68 % | 0.03 / 0.01 | 105 / 30 | 0 % / 0 % | 22 / 17 | 0 / 0 |
| 62426694 | `8716807#6_0` | 59 | 85 % | 0.22 / 0.55 | 75 / 225 | 0 % / 1 % | 32 / 20 | 0 / 0 |
| 280120513 | `297047310#4_0` | 55 | 91 % | 0.08 / 0.03 | 120 / 15 | 0 % / 0 % | 48 / 18 | 0 / 0 |
| 280120513 | `-23648008#0_0` | 76 | 66 % | 0.12 / 0.16 | 120 / 480 | 0 % / 20 % | 58 / 20 | 0 / 0 |
| 280120513 | `-28675493_0` | 91 | 55 % | 0.21 / 0.08 | 30 / 0 | 0 % / 0 % | 118 / 17 | 15 / 0 |
| 280120513 | `-28675493_1` | 91 | 55 % | 0.00 / 0.00 | 120 / 15 | 0 % / 0 % | 12 / 0 | 0 / 0 |
| 256201389 | `-24487264_0` | 166 | 30 % | 0.00 / 0.00 | 210 / 3600 | 0 % / 95 % | 0 / 0 | 0 / 0 |
| 256201389 | `-225249129#0_0` | 13 | 100 % | 0.02 / 0.00 | 0 / 0 | 0 % / 0 % | 45 / 0 | 0 / 0 |
| 256201389 | `23648008#2_0` | 176 | 28 % | 0.02 / 0.00 | 30 / 0 | 0 % / 0 % | 21 / 0 | 0 / 0 |
| 26110729 | `-42925825#2_0` | 254 | 20 % | 0.85 / 1.36 | 90 / 105 | 0 % / 0 % | 46 / 56 | 0 / 0 |
| 26110729 | `186623965#9_0` | 160 | 31 % | 0.15 / 0.12 | 120 / 90 | 0 % / 0 % | 72 / 45 | 3 / 0 |
| 26110729 | `186623965#9_1` | 160 | 31 % | 0.14 / 0.11 | 120 / 90 | 0 % / 0 % | 59 / 29 | 0 / 0 |
| 26110729 | `-297047310#2_0` | 601 | 8 % | 0.40 / 0.38 | 90 / 105 | 0 % / 0 % | 76 / 20 | 6 / 0 |
| 26110729 | `-186623965#16_0` | 188 | 27 % | 0.35 / 0.38 | 120 / 90 | 0 % / 0 % | 85 / 76 | 10 / 3 |
| 26110729 | `-186623965#16_1` | 188 | 27 % | 0.33 / 0.37 | 120 / 90 | 0 % / 0 % | 82 / 54 | 9 / 0 |
| 247379907 | `22917421#3_0` | 96 | 52 % | 0.48 / 0.38 | 495 / 240 | 8 % / 2 % | 493 / 21 | 153 / 0 |
| 247379907 | `186623965#15_0` | 188 | 27 % | 0.15 / 0.07 | 120 / 30 | 0 % / 0 % | 99 / 24 | 13 / 0 |
| 247379907 | `186623965#15_1` | 188 | 27 % | 0.11 / 0.06 | 75 / 30 | 0 % / 0 % | 102 / 23 | 10 / 0 |
| 247379907 | `-22917421#14_0` | 534 | 9 % | 0.39 / 0.38 | 495 / 240 | 8 % / 2 % | 465 / 20 | 447 / 0 |
| 247379907 | `-186623965#18_0` | 145 | 35 % | 0.24 / 0.16 | 120 / 30 | 0 % / 0 % | 67 / 27 | 2 / 0 |
| 247379907 | `-186623965#18_1` | 145 | 35 % | 0.23 / 0.16 | 75 / 30 | 0 % / 0 % | 61 / 28 | 0 / 0 |
| cluster_10... | `22917421#5_0` | 533 | 9 % | 0.17 / 0.28 | 90 / 255 | 0 % / 3 % | 75 / 71 | 7 / 2 |
| cluster_10... | `28675510#4_0` | 89 | 56 % | 0.25 / 0.14 | 60 / 30 | 0 % / 0 % | 41 / 31 | 0 / 0 |
| cluster_10... | `-22959475#4_0` | 191 | 26 % | 0.00 / 0.00 | 90 / 255 | 0 % / 3 % | 0 / 0 | 0 / 0 |
| cluster_10... | `-28675510#11_0` | 258 | 19 % | 0.16 / 0.24 | 60 / 30 | 0 % / 0 % | 29 / 36 | 0 / 0 |

## Phase use (share of decisions)

| Junction | ARLGOS-v2 | Greedy | Time in the subset phase A / G | Halting per lane during it A / G |
|---|---|---|---|---|
| 252017285 | 50% 50% | 60% 40% | - | - |
| 32319828 | 100% 0% | 100% 0% | 0 % / 0 % | - / - |
| 62426694 | 39% 14% 47% | 69% 0% 31% | - | - |
| 280120513 | 62% 7% 31% | 85% 0% 15% | - | - |
| 256201389 | 44% 25% 31% | 100% 0% 0% | 17 % / 0 % | 0.02 / - |
| 26110729 | 47% 2% 51% 0% | 55% 0% 45% 0% | 0 % / 0 % | 0.37 / - |
| 247379907 | 61% 7% 32% 0% | 65% 0% 35% 0% | 0 % / 0 % | 0.67 / - |
| cluster_10... | 58% 0% 41% 1% | 81% 0% 19% 0% | 1 % / 0 % | 0.18 / - |

## Phase strings of the junctions with aliased phases

* `32319828`: 0 `GGggGGgg` lanes [1.0, 1.0], 1 `rrGGrrGG` lanes [1.0, 1.0]
* `256201389`: 0 `rrrGGgGgg` lanes [0.0, 1.0, 1.0], 1 `rrrrrGrGG` lanes [0.0, 1.0, 1.0], 2 `GGgGrrrrr` lanes [1.0, 1.0, 0.0]
* `26110729`: 0 `rrrrGGGggrrrrGGGgg` lanes [0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 1 `rrrrrrrGGrrrrrrrGG` lanes [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2 `GGggrrrrrGGggrrrrr` lanes [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3 `rrGGrrrrrrrGGrrrrr` lanes [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
* `247379907`: 0 `rrrrGGGggrrrrGGGgg` lanes [0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 1 `rrrrrrrGGrrrrrrrGG` lanes [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2 `GGggrrrrrGGggrrrrr` lanes [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3 `rrGGrrrrrrrGGrrrrr` lanes [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
* `cluster_1098574052_1098574061_247379905`: 0 `rrrrGGggrrrrGGgg` lanes [0.0, 1.0, 0.0, 1.0], 1 `rrrrrrGGrrrrrrGG` lanes [0.0, 1.0, 0.0, 1.0], 2 `GGggrrrrGGggrrrr` lanes [1.0, 0.0, 1.0, 0.0], 3 `rrGGrrrrrrGGrrrr` lanes [1.0, 0.0, 1.0, 0.0]

## The same audit on every network config in the fork (static)

| Network | Junctions | Phases | Aliased pairs (max) | Junctions with one (max) | Aliased pairs (mean) | action_space differs from valid_phase_list |
|---|---|---|---|---|---|---|
| arterial_network_4_4 | 16 | 80 | 0 | 0 | 0 | 0 |
| cologne_network_8 | 8 | 25 | 6 | 5 | 0 | 0 |
| corridor_3 | 3 | 6 | 0 | 0 | 0 | 0 |
| grid_network_4_4 | 16 | 128 | 0 | 0 | 0 | 0 |
| grid_network_5_5 | 25 | 125 | 0 | 0 | 0 | 0 |
| ingolstadt_network_21 | 21 | 66 | 0 | 0 | 0 | 2 |
| monaco_network_30 | 28 | 94 | 20 | 12 | 0 | 0 |
| shaoxing_network_7 | 7 | 25 | 0 | 0 | 0 | 0 |
| shenzhen_network_29 | 29 | 59 | 0 | 0 | 0 | 0 |
| shenzhen_network_55 | 55 | 114 | 0 | 0 | 0 | 0 |
