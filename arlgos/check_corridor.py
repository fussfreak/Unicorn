"""ARLGOS (T9): self-checks of the corridor port. arlgos/check_fork.py runs this file in a
subprocess with UNICORN_NET=corridor_3 (parameters.py reads it at import).

    python arlgos/check_corridor.py

Checks that
  1. parameters.py reserves the corridor's seeds (test 1042-1061, G2 1062-1161, val 5042-5061)
     and draw_train_seed() never returns one;
  2. maps/corridor_3 holds ARLGOS's scenario (fingerprint 73ac41067b5b) and its config numbers
     the movements in SUMO's link order, with action 0 = EW green and 1 = NS green;
  3. the SignalController port keeps ARLGOS's rules: 60 s warm-up greens, 10 s yellow, 5 s
     all-red, a green ends at its target (30-90 s), and next_green() follows the cycle;
  4. Unicorn's action maps to keep (90 s) / switch (30 s) in a green, transitions are forced
     through the phase mask, and both sensing modes give 14 x 8 observations ('camera' with
     moving / occupancy features zero);
  5. arlgos/corridor_parity.py's last report says PASS for this scenario;
  6. env/sumo_api.py gives traci by default and libsumo with UNICORN_LIBSUMO=1 (where installed);
  7. arlgos/watch_corridor.py (sumo-gui, for looking) picks a finished numbered snapshot, never the
     resume file model/checkpoint.pkl, and adds only GUI arguments to CorridorEnv's;
  8. Unicorn's policy draws VAE noise in every forward pass, so arlgos/eval_corridor.py seeds torch
     for every episode (greedy too); its validation is saved after every episode;
  9. arlgos/run_T10.ps1 retries a crashed run (a fake run that crashes twice, then succeeds) and
     gives up after -Retries (a fake run that always crashes); -NoCleanup, so the test never
     touches a real training run's processes.
Exits 1 if any check fails.
"""
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
ARLGOS_FINGERPRINT = '73ac41067b5b'       # ARLGOS sumo_env.scenario_fingerprint(), frozen since 2026-11-01
FAILED = []


def check(name, ok, detail=''):
    print('[{}] {}{}'.format('PASS' if ok else 'FAIL', name, ': ' + str(detail) if detail else ''))
    if not ok:
        FAILED.append(name)


def main():
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    sys.path.insert(0, HERE)
    if os.environ.get('UNICORN_NET') != 'corridor_3':
        raise SystemExit('run with UNICORN_NET=corridor_3 (arlgos/check_fork.py does)')
    import numpy as np
    from common import use_env_sumo
    use_env_sumo()
    from parameters import SUMO_PARAMS

    # 1. seeds
    check('corridor TEST_SEEDS = 1042-1161 (test + G2), VAL_SEEDS = 5042-5061',
          SUMO_PARAMS.TEST_SEEDS == list(range(1042, 1162)) and SUMO_PARAMS.VAL_SEEDS == list(range(5042, 5062)))
    from env import matsc
    np.random.seed(1)
    reserved = set(SUMO_PARAMS.TEST_SEEDS) | set(SUMO_PARAMS.VAL_SEEDS)
    hits = sum(matsc.draw_train_seed() in reserved for _ in range(20000))
    check('20000 corridor training seeds, none reserved', hits == 0, '{} hits'.format(hits))

    # 2. map and config
    from env.corridor import CorridorEnv, scenario_fingerprint, PHASE_OF_ACTION, sumo_binary, CFG
    from env.corridor_signals import (SignalController, STATES, EW_GREEN, NS_GREEN, EW_YELLOW, NS_YELLOW,
                                      ALL_RED, FIXED_DELTA_SECONDS)
    fp = scenario_fingerprint()
    check('maps/corridor_3 = ARLGOS scenario {}'.format(ARLGOS_FINGERPRINT), fp == ARLGOS_FINGERPRINT, fp)
    env = CorridorEnv(None, test=True, sensing='full')
    ok = env.tls_list == ['C3', 'C2', 'C1'] and all(
        len(env.tls_dict[t].traffic_movement_list) == 14 and
        env.tls_dict[t].action_space == [env.state_strings[t][('RED', 'GREEN')], env.state_strings[t][('GREEN', 'RED')]]
        for t in env.tls_list) and PHASE_OF_ACTION == (EW_GREEN, NS_GREEN)
    check('3 junctions x 14 movements; action_space = [EW green, NS green]', ok)
    from env.sumo_api import traci
    traci.start([sumo_binary('sumo'), '-c', CFG, '--no-warnings', 'true'])
    try:
        order_ok = all([tuple(l[0][:2]) for l in traci.trafficlight.getControlledLinks(t)]
                       == list(env.tls_dict[t].traffic_movement_list) for t in env.tls_list)
    finally:
        traci.close()
    check("Unicorn's movement i = SUMO link i at every junction", order_ok)

    # 3. the SignalController port
    c = SignalController('x')
    seen, t = [], 0.0
    for _ in range(int(round(300 / FIXED_DELTA_SECONDS))):
        if not seen or seen[-1][0] != c.phase:
            seen.append((c.phase, round(t, 2)))
        c.step(FIXED_DELTA_SECONDS)
        t += FIXED_DELTA_SECONDS
    starts = [s for _, s in seen]
    want = [(EW_GREEN, 0.0), (EW_YELLOW, 60.0), (ALL_RED, 70.0), (NS_GREEN, 75.0), (NS_YELLOW, 135.0),
            (ALL_RED, 145.0), (EW_GREEN, 150.0)]
    check('fixed-time cycle: 60 s green, 10 s yellow, 5 s all-red, 150 s cycle', seen[:7] == want, seen[:7])
    c = SignalController('x')
    for _ in range(int(round(35 / FIXED_DELTA_SECONDS))):
        c.step(FIXED_DELTA_SECONDS)
    c.set_target(30)
    c.step(FIXED_DELTA_SECONDS)
    ends_now = c.phase == EW_YELLOW
    c2 = SignalController('y')
    c2.set_target(90)
    n = 0
    while c2.phase == EW_GREEN:
        c2.step(FIXED_DELTA_SECONDS)
        n += 1
    check('switch past 30 s ends the green on the next tick; keep holds it to 90 s',
          ends_now and abs(n * FIXED_DELTA_SECONDS - 90) < 1e-6, (ends_now, n * FIXED_DELTA_SECONDS))
    c3, nxt = SignalController('z'), []
    for _ in range(int(round(150 / FIXED_DELTA_SECONDS))):
        nxt.append((c3.phase, c3.next_green()))
        c3.step(FIXED_DELTA_SECONDS)
    expect = {EW_GREEN: EW_GREEN, EW_YELLOW: NS_GREEN, NS_GREEN: NS_GREEN, NS_YELLOW: EW_GREEN}
    ok = all(g == expect[p] for p, g in nxt if p != ALL_RED) and \
        {g for p, g in nxt if p == ALL_RED} == {NS_GREEN, EW_GREEN}
    check('next_green() follows the cycle (yellow/all-red -> the other green)', ok)

    # 4. actions, masks, observations
    env.signals.reset()
    tl = env.tls_list
    keep = env.targets_for({tl[0]: 0, tl[1]: 0, tl[2]: 0})
    switch = env.targets_for({tl[0]: 1, tl[1]: 1, tl[2]: 1})
    for _ in range(int(round(65 / FIXED_DELTA_SECONDS))):
        env.signals.step(FIXED_DELTA_SECONDS)       # all in EW yellow now
    trans = env.targets_for({tl[0]: 0, tl[1]: 1, tl[2]: 0})
    forced = env.forced_actions()
    _, masks = env.get_phase_vec_mask_dict()
    check('EW green: action 0 keeps (90 s), action 1 switches (30 s); in yellow 90 s arms the next green',
          keep == [90.0] * 3 and switch == [30.0] * 3 and trans == [90.0] * 3, (keep, switch, trans))
    check('in EW yellow the mask leaves only NS green (action 1)',
          forced == {t: 1 for t in tl} and all(list(masks[t]) == [1, 0] for t in tl), (forced, masks))
    for sensing in ('full', 'camera'):
        e = CorridorEnv(None, test=True, sensing=sensing)
        obs = e.start_episode(424242)              # not a reserved seed
        for _ in range(3):
            obs, rew, done, info = e.step({t: 0 for t in e.tls_list})
        _, m = e.get_phase_vec_mask_dict()
        shapes = {t: obs[t].shape for t in obs}
        ok = all(s == (14, 8) for s in shapes.values()) and np.isfinite(list(rew.values())).all() \
            and len(e.aql_series) == 4 and not done
        if sensing == 'camera':
            ok = ok and all((obs[t][:, 3:7] == 0).all() for t in obs) and any(obs[t][:, 1].sum() > 0 for t in obs)
        check('{} sensing: 3 decisions run, 14 x 8 observations{}'.format(
            sensing, ', moving/occupancy zero' if sensing == 'camera' else ''), ok, shapes)
        e.close()

    # 5. the cross-environment parity report
    report = os.path.join(FORK, 'results', 'T9', 'parity.md')
    text = open(report, encoding='utf-8').read() if os.path.exists(report) else ''
    check('arlgos/corridor_parity.py: last report PASS for scenario {}'.format(ARLGOS_FINGERPRINT),
          'PASS: identical' in text and ARLGOS_FINGERPRINT in text,
          'missing: run arlgos/parity.ps1' if not text else '')

    # 6. SUMO client switch
    code = 'from env.sumo_api import traci; print(traci.__name__)'
    names = {}
    for flag in ('', '1'):
        out = subprocess.run([sys.executable, '-c', code], cwd=FORK, capture_output=True, text=True,
                             env=dict(os.environ, UNICORN_LIBSUMO=flag))
        names[flag] = out.stdout.strip() or 'unavailable'
    has_libsumo = importlib.util.find_spec('libsumo') is not None
    check('sumo_api: traci by default, libsumo with UNICORN_LIBSUMO=1' +
          ('' if has_libsumo else ' (libsumo not installed here: default only)'),
          names[''] == 'traci' and (names['1'] == 'libsumo' if has_libsumo else True), names)

    # 7. watching: a finished snapshot, never the resume file; only GUI arguments added
    import re
    import tempfile
    import time
    import torch
    import eval_corridor
    import watch_corridor as wc
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'model'))
        for n, age in ((50, 600), (100, 300), (150, 2)):           # 150: still being written
            p = os.path.join(d, 'model', 'checkpoint{}.pkl'.format(n))
            open(p, 'w').close()
            os.utime(p, (time.time() - age, time.time() - age))
        open(os.path.join(d, 'model', 'checkpoint.pkl'), 'w').close()
        latest, _ = wc.pick_checkpoint(d, 'latest')
        auto, _ = wc.pick_checkpoint(d, 'auto')
    check('watch_corridor: latest = newest finished snapshot (checkpoint100), never checkpoint.pkl',
          os.path.basename(latest) == 'checkpoint100.pkl' and auto == latest,
          (os.path.basename(latest), os.path.basename(auto)))
    gui = wc.gui_env_class(CorridorEnv, None, None, 50)(None, test=True)
    ga, ca = gui.sumo_args(424242), CorridorEnv(None, test=True).sumo_args(424242)
    check('watch_corridor: sumo-gui gets the scenario arguments plus --start --quit-on-end --delay 0',
          ga == ca + ['--start', '--quit-on-end', '--delay', '0'], ga[len(ca):])

    # 8. VAE noise: unseeded forward passes differ, seeded ones match; eval seeds every episode
    from models.Unicorn import Unicorn
    e = CorridorEnv(None, test=True, sensing='full')
    obs = e.start_episode(424242)
    pv, mask = e.get_phase_vec_mask_dict()
    ia = e.get_int_attr_vec_dict()
    e.close()

    def t(dd):
        return torch.tensor(np.array(list(dd.values())), dtype=torch.float32)
    net = Unicorn(input_dim=e.tls_obs_space, agent_dim=e.tls_agent_space, int_vec_dim=e.tls_int_attr_space,
                  actor_lr=1e-4, critic_lr=1e-4)
    net.eval()

    def policy(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            return net(t(obs), t(pv), t(ia), t(mask), None)[0]
    a, b, c1, c2 = policy(), policy(), policy(7), policy(7)
    import inspect
    src = inspect.getsource(eval_corridor.play)
    every = re.search(r'^    torch\.manual_seed\(SAMPLE_SEED \+ seed\)', src, re.M) is not None
    check("Unicorn's policy draws VAE noise (unseeded passes differ, seeded match); eval seeds every episode",
          (not torch.equal(a, b)) and torch.equal(c1, c2) and every,
          {'unseeded differ': not torch.equal(a, b), 'seeded match': torch.equal(c1, c2), 'eval seeds all': every})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'val.csv')
        save = eval_corridor.save_progress(p, [])
        row = {'checkpoint': 'checkpoint50.pkl', 'seed': 5042, 'aql': 3.1}
        save([row])
        save([row, dict(row, seed=5043)])
        back = [r['seed'] for r in eval_corridor.read_csv(p)]
        check('eval_corridor: validation saved after every episode (whole file replaced, no .tmp left)',
              back == ['5042', '5043'] and not os.path.exists(p + '.tmp'), back)

    # 9. run_T10.ps1 retries a crashed run, and gives up after -Retries
    fake = ("param([string]$Sensing, [int]$Seed, [string]$Python)\n"
            "$f = Join-Path $PSScriptRoot 'attempts.txt'\n"
            "$n = 1 + [int](Get-Content $f -ErrorAction SilentlyContinue)\n"
            "Set-Content $f $n\n"
            "if ($n -le [int](Get-Content (Join-Path $PSScriptRoot 'crashes.txt'))) { throw \"fake crash $n\" }\n")

    def t10(crashes, retries):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'fake_run.ps1'), 'w') as f:
                f.write(fake)
            with open(os.path.join(d, 'crashes.txt'), 'w') as f:
                f.write(str(crashes))
            out = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                  os.path.join(HERE, 'run_T10.ps1'), '-Seeds', '1', '-Sensing', 'full',
                                  '-Runner', os.path.join(d, 'fake_run.ps1'), '-Log', os.path.join(d, 'q.log'),
                                  '-Retries', str(retries), '-RetryWaitSec', '0', '-NoCleanup'],
                                 capture_output=True, text=True)
            with open(os.path.join(d, 'attempts.txt')) as f:
                attempts = int(f.read().strip())
            with open(os.path.join(d, 'q.log')) as f:
                log = f.read()
        return out.returncode, attempts, log
    rc, n, log = t10(2, 3)
    ok1 = (rc == 0 and n == 3 and log.count('crashed corridor3_full_seed1') == 2
           and '(retry 2 of 3)' in log and 'done  corridor3_full_seed1' in log)
    rc2, n2, log2 = t10(99, 2)
    ok2 = rc2 != 0 and n2 == 3 and 'FAILED corridor3_full_seed1 after 2 retries' in log2
    check('run_T10: a run that crashes twice is retried and finishes; one that always crashes stops after -Retries',
          ok1 and ok2, {'crash x2': (rc, n), 'always': (rc2, n2)})

    print('\n{} failed'.format(len(FAILED)) if FAILED else '\nall corridor checks passed')
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
