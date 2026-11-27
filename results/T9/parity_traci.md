# T9 parity: the fork's corridor env vs ARLGOS's SumoEnv

Seeds 1042, 1043, 1044; policies fixed-time 60 s and a fixed keep/switch sequence (switch probability 0.3). Fork: traci (Python 3.8.20). ARLGOS: libsumo (env arlgos_sumo). Scenario fingerprint 73ac41067b5b.

| Episode | What | Result |
|---|---|---|
| fixed60/1042 | demand (departure times and ids) | same |
| fixed60/1042 | light strings, every tick | same |
| fixed60/1042 | camera counts at t = 0 | same |
| fixed60/1042 | AQL series | same |
| fixed60/1042 | T7 metrics (max abs diff 0) | same |
| fixed60/1043 | demand (departure times and ids) | same |
| fixed60/1043 | light strings, every tick | same |
| fixed60/1043 | camera counts at t = 0 | same |
| fixed60/1043 | AQL series | same |
| fixed60/1043 | T7 metrics (max abs diff 0) | same |
| fixed60/1044 | demand (departure times and ids) | same |
| fixed60/1044 | light strings, every tick | same |
| fixed60/1044 | camera counts at t = 0 | same |
| fixed60/1044 | AQL series | same |
| fixed60/1044 | T7 metrics (max abs diff 0) | same |
| keepswitch/1042 | demand (departure times and ids) | same |
| keepswitch/1042 | light strings, every tick | same |
| keepswitch/1042 | camera counts at t = 0 | same |
| keepswitch/1042 | AQL series | same |
| keepswitch/1042 | T7 metrics (max abs diff 0) | same |
| keepswitch/1043 | demand (departure times and ids) | same |
| keepswitch/1043 | light strings, every tick | same |
| keepswitch/1043 | camera counts at t = 0 | same |
| keepswitch/1043 | AQL series | same |
| keepswitch/1043 | T7 metrics (max abs diff 0) | same |
| keepswitch/1044 | demand (departure times and ids) | same |
| keepswitch/1044 | light strings, every tick | same |
| keepswitch/1044 | camera counts at t = 0 | same |
| keepswitch/1044 | AQL series | same |
| keepswitch/1044 | T7 metrics (max abs diff 0) | same |

observe_full equals Unicorn's Tls.observe_resco_network at 654 junction-decisions in green: 0 mismatches.

**PASS: identical**
