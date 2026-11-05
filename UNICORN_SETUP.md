# Unicorn fork for ARLGOS: setup and changes

This is the [ARLGOS](https://github.com/fussfreak/ARLGOS) fork of
[Unicorn](https://github.com/marmotlab/Unicorn) (Zhang et al., arXiv 2503.11488). It is the
baseline for the beat-Unicorn plan (ARLGOS `ROADMAP.md`).

* Branch `arlgos` holds our changes; `main` stays identical to upstream.
* Remotes: `origin` = `fussfreak/Unicorn`, `upstream` = `marmotlab/Unicorn`.

Set up 2026-11-03 on the laptop (task T3): Windows 11, i5-1135G7 (4 cores, 8 threads),
7.7 GB RAM. It runs natively on Windows, Ray included; WSL isn't needed.

## 1. Commands

Run these in PowerShell, from the repo root (`E:\ARLGOS\Unicorn`). `conda` isn't on PATH,
so the commands call it by its full path.

```powershell
# environments (once, about 15 min each)
C:\Users\DELL\miniconda3\Scripts\conda.exe env create -f environments\unicorn_cpu.yml       # "unicorn": training and evaluation
C:\Users\DELL\miniconda3\Scripts\conda.exe env create -f environments\unicorn_sumo116.yml   # only for the SUMO version check

$env:PYTHONPATH = ''        # the standalone SUMO installer puts its tools folder here; it would shadow the env's traci
$py = "C:\Users\DELL\miniconda3\envs\unicorn\python.exe"

& $py arlgos\check_fork.py                                    # self-checks: run after every change to the fork
& C:\Users\DELL\miniconda3\envs\unicorn_sumo116\python.exe arlgos\check_fork.py --expect-sumo 1.16.0

.\arlgos\run_unicorn.ps1 -Episodes 4 -Name smoke_cologne8     # smoke run: 3 PPO updates x 6 workers
& $py arlgos\timing.py --name smoke_cologne8 --out results\T3\smoke_timing

.\arlgos\version_check.ps1                                    # SUMO 1.16 vs 1.28 -> results\T3\version_check.md

.\arlgos\run_unicorn.ps1 -Seed 1                              # a full run: 1500 "episodes" (T4)
tensorboard --logdir Train_MATSC                              # watch training
```

The launchers set what upstream expected you to do by hand:
* `SUMO_HOME` points at the env's own `eclipse-sumo`, so the SUMO version belongs to the env.
* `PYTHONPATH` is cleared.
* Per-run settings go in `UNICORN_*` environment variables (section 3).
* Windows is kept awake, and every output line is timestamped into `logs\`.

**Firewall:** the first time SUMO or Ray opens a local port, Windows may ask whether
`sumo.exe` or `python.exe` may use the network. Allow it on private networks.

## 2. Environment

| | Upstream | Here | Why |
|---|---|---|---|
| Python | ≥ 3.8 (conda env 3.8) | 3.8.20 | as upstream |
| PyTorch | 1.13.0 + CUDA 11.7 | **1.13.0+cpu** | The lab PC's RTX 5060 Ti (Blackwell) needs CUDA 12.8 and PyTorch ≥ 2.7, so the pinned build can't use it. SUMO dominates the run time anyway. |
| SUMO | "≥ 1.16.0" (simulator not pinned); `traci`/`sumolib` 1.16.0 | **`eclipse-sumo`, `traci`, `sumolib` 1.28.0** | ARLGOS uses 1.28.0, and head-to-heads need one simulator for both methods. The client matches the simulator (TraCI API 22). |
| Everything else | `requirements.txt` | same pins | |

`environments/unicorn_sumo116.yml` is the same env with SUMO 1.16.0. It is used only by
`arlgos/version_check.ps1`. On Python 3.8, `libsumo` 1.28 has no wheel, but Unicorn talks to
SUMO through `traci` (a socket), so it doesn't need `libsumo`.

## 3. Changes from upstream

Every changed line is marked `# ARLGOS:`. Run `git diff main -- '*.py'` to see them all.

| File | Change | Why |
|---|---|---|
| `parameters.py` | `NET_NAME` = `cologne_network_8` (env var `UNICORN_NET`) | T3/T4 target network; upstream defaulted to Grid 5×5 |
| `parameters.py` | Green / yellow / teleport set from the dataset family: RESCO 15/5/−1, MA2C 10/3/300, GESA 15/5/600 | README "Recommended settings" and paper §V-C. Upstream hard-coded 15/5/300, and its comment says something else again. |
| `parameters.py` | `MAX_EPISODES` 1500 (`UNICORN_MAX_EPISODES`); upstream 3000 | The paper: "around 1500 episodes" |
| `parameters.py` | `NUM_META_AGENTS` can be overridden (`UNICORN_WORKERS`); default 6 as upstream | lets the laptop fall back to fewer workers |
| `parameters.py` | `EXPERIMENT_NAME` can be overridden (`UNICORN_EXP_NAME`) | upstream's name (date + hour) collides when you start two runs in the same hour |
| `parameters.py` | `USE_GPU = False`, `NUM_GPU = 0` | CPU torch |
| `parameters.py` | `SAVE_MODEL_STEP` 50 (upstream 500) | so a checkpoint can later be chosen on validation seeds |
| `parameters.py` | `TRAIN_SEED` (`UNICORN_SEED`, default 0), `TEST_SEEDS`, `VAL_SEEDS` | see the next three rows |
| `driver_unicorn.py` | seeds `random` / numpy / torch with `TRAIN_SEED` instead of 0 | Different training seeds become possible. Default 0 = upstream. |
| `runner_unicorn.py` | seeds worker *k* with `TRAIN_SEED·100 + k` (numpy and torch) | Upstream never seeded the workers, so their episode seeds and sampled actions came from OS entropy and no run was repeatable. |
| `env/matsc.py` | training episodes draw their SUMO seed through `draw_train_seed()`, which redraws the 10 test seeds (1000, 1100, …, 1900) and the validation seeds 5042–5061 | Upstream drew from 0–99 999. Over the ~9000 training simulations of a full run, that gives a ~59 % chance of training on a test seed. |
| `.gitignore` | run outputs and the per-worker files SUMO writes next to each map | |
| `utils.py`, `driver_unicorn.py` | `save_latest` writes `model/checkpoint.pkl` (weights, both Adam states, the counter, the `random` / numpy / torch states) after **every** update, atomically; `load_latest` restores it. The driver resumes from it, and does not add one to the counter again on resume. | T4: upstream's `LOAD_MODEL` loads `checkpoint.pkl`, which nothing wrote (section 4, item 5). A multi-day run must survive a reboot. |
| `runner_unicorn.py` | workers get the start episode; a resumed run seeds them with `[TRAIN_SEED·100 + k, start]`, a fresh run as before. The resume path is joined with `os.path.join`. | A resumed run would otherwise replay its first training seeds. Upstream's path lacked a `/`. |
| `parameters.py` | `LOAD_MODEL` = (`UNICORN_RESUME` == `1`), `EXPERIMENT_PATH` = the run named by `UNICORN_EXP_NAME` | `run_unicorn.ps1 -Resume` sets both |

Added: `environments/`, `arlgos/` (launchers and checks), this file, and `results/` (summaries only).

## 4. Things found in the upstream code that we did NOT change

These are part of the published method. Changing them would mean we no longer run "Unicorn".

1. **GAE λ is 0.95 in the code, 0.98 in the paper.** `parameters.py` defines `LAMBDA_ADV = .98`,
   but nothing reads it. `runner_unicorn.py:188` computes advantages with `gamma * 0.95`, i.e. λ = 0.95.
   We run the code as published. `check_fork.py` checks that this line is unchanged.
2. **Advantage normalisation has a precedence slip** (`driver_unicorn.py:115`):
   `adv - adv.mean() / (adv.std() + 1e-5)` subtracts mean/std instead of computing `(adv - mean) / std`.
3. **`CONTRASTIVE_LOSS = False` is not read anywhere.** The contrastive (NT-Xent) loss is always on,
   as the paper describes.
4. **One "episode" in the training counter = one PPO update = one 3600 s simulation on each of the
   6 workers.** So 1500 "episodes" is about 9000 simulated hours. TensorBoard's x-axis is this counter.
5. **Resuming is broken.** `LOAD_MODEL` loads `model/checkpoint.pkl` (`driver_unicorn.py:297`), but
   only `checkpoint{N}.pkl` is ever saved. The workers load `EXPERIMENT_PATH + 'model/state_dict.pth'`,
   which is missing a `/` (`runner_unicorn.py:214`). **Fixed on 2026-11-03 (T4)**, see section 3
   and `run_unicorn.ps1 -Resume`. A resumed run loses at most the update in progress; its workers
   draw new training seeds, so it is valid but not bit-identical to an uninterrupted run.
6. **The RESCO observation uses 50 m, not 200 m.** `env/tls.py: observe_resco_network`'s docstring
   says 200 m detectors, but the code counts vehicles within `distance = 50` m. On outgoing lanes it
   measures the 50 m *before the next stop line*, i.e. the far end of the lane.
7. **`evaluator_non_rl.py` imports `os` only under `if __name__ == '__main__'`.** Importing its
   `Evaluator` from elsewhere needs `evaluator_non_rl.os = os`, which `arlgos/version_check.py` sets.

## 5. Results

* `results/T3/smoke_timing.md`: 3 updates × 6 workers on Cologne8 completed. **115 s per update
  on the laptop, so about 2.0 days per 1500-episode run.**
* `results/T3/version_check.md`: SUMO 1.16 vs 1.28 on Cologne8, with Unicorn's own non-RL
  controllers on its 10 test seeds. **Within ±5 % on every metric.** The Grid 5×5 setup check
  against the paper's Table II ran out of RAM on the laptop. It moves to the lab PC (T4):
  `.\arlgos\version_check.ps1 -Nets grid_network_5_5`.
