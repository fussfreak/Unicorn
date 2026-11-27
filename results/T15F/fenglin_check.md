# Fenglin switches: checks (T15F)

From `arlgos/fenglin_check.py`; 13 of 13 passed.

| Check | Result | Detail |
|---|---|---|
| cfg_upstream | pass | UNICORN_FCD=1 sumocfg identical to HEAD's (743 bytes) |
| cfg_default | pass | default = upstream minus 2 lines: ['<fcd-output value="fcd.xml"/>', '<fcd-output.signals value="true"/>'] |
| launcher_running_count | pass | run_arlgos.ps1 counts running seeds as @(Running $s).Count (2 places, 0 unwrapped) |
| heldout_base | pass | 6677 vehicles with unique ids in fenglin_y2z_t.rou.xml |
| heldout_deterministic | pass | seed 5042 twice -> identical day |
| heldout_seeds_differ | pass | seed 5042 vs 1000: 6663 of 6677 departure times differ |
| heldout_valid | pass | keep=1: 6677 vehicles, sorted, 0.00-3599.00 s, routes unchanged, max shift 59.98 s, 0 beyond jitter |
| heldout_jitter_spread | pass | mean |shift| 29.7 s (uniform on [-60, 60] gives 30) |
| heldout_keep | pass | keep=0.9: 6009 of 6677 kept (expected 6009 +- 98); kept vehicles at the keep=1 times: True |
| fcd_equal_900201 | pass | fcd on vs off: 3600 traffic rows, 4259 trips, queue 1.4653; differing: nothing; 99.7 s vs 65.5 s |
| fcd_equal_900202 | pass | fcd on vs off: 3600 traffic rows, 4273 trips, queue 1.3384; differing: nothing; 95.9 s vs 63.3 s |
| val_score_same | pass | eval_unicorn.decision_queue vs ArlgosEnv.queue_series at all 1205 decision points of 5 episodes: max difference 0.0 |
| heldout_sim | pass | published loaded 6677 / inserted 4943; held-out loaded 6677 / inserted 5211; traffic differs: True |

| Episode (seed / fcd / demand) | Client | Seconds | Decision-point queue | Loaded | Inserted |
|---|---|---|---|---|---|
| 900201 / 1 / published | libsumo | 99.7 | 1.4653 | 6677 | 4943 |
| 900201 / 0 / published | libsumo | 65.5 | 1.4653 | 6677 | 4943 |
| 900202 / 1 / published | libsumo | 95.9 | 1.3384 | 6677 | 4945 |
| 900202 / 0 / published | libsumo | 63.3 | 1.3384 | 6677 | 4945 |
| 900201 / 0 / heldout | libsumo | 61.5 | 1.5267 | 6677 | 5211 |
