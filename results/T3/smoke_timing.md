# Unicorn training cost: smoke_cologne8 (cologne_network_8)

Machine: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 logical CPUs, 7.7 GB RAM), Python 3.8.20, torch 1.13.0+cpu, Ray 2.3.1, SUMO 1.28.0. Fork commit 1d2b0b8+uncommitted.

One update = one 3600 s episode on each of the 6 workers, then the PPO update.

| | Value |
|---|---|
| Updates timed | 3 (first one includes Ray start-up: 179 s) |
| **Seconds per update** | **115** (gaps: 126, 105) |
| One worker episode (simulation + acting) | mean 78 s, max 95 s, n = 18 |
| Learner (PPO update + waiting) per update | 40 s |
| RAM used (whole machine) | before 7017 MB, peak 7438 MB of 7926 MB |
| **Full run (1500 episodes = 1499 updates = 8994 simulations)** | **48 h = 2.0 days** |

The full-run figure assumes every update costs what the timed ones did; it
holds only for this machine and this number of workers.
