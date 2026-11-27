"""ARLGOS (T15F): checks for the two Fenglin switches in maps/shaoxing_network_7/build_file.py.

    python arlgos/fenglin_check.py                 # every check -> results/T15F/fenglin_check.{md,json}
    python arlgos/fenglin_check.py --quick         # the checks that need no simulation

What is checked:
  cfg_upstream   UNICORN_FCD=1 (published demand) writes upstream's sumocfg byte for byte
                 (compared with the generator in the fork's last commit, `git show HEAD:...`).
  cfg_default    the default sumocfg is upstream's minus exactly the two fcd-output lines.
  heldout_*      the held-out traffic days: deterministic per seed, different across seeds, every
                 vehicle kept with its route (keep = 1), departures sorted, inside [0, 3599] s and
                 within the jitter of the original time; thinning (keep < 1) removes about the
                 expected share and does not move the vehicles it keeps.
  fcd_equal      SIMULATION: two seeds, the same seeded random phase sequence, fcd.xml on vs off:
                 every per-second traffic row, every trip and the decision-point queue are
                 identical. Also times one 3600 s episode each way.
  heldout_sim    SIMULATION: a held-out day loads and inserts all 6677 vehicles, as the published
                 day does, and its traffic differs from the published day of the same seed.

The simulation checks run 5 one-hour episodes as separate processes in parallel (each with its
own TraCI port and files), with libsumo when the env has it. Seeds 900201-900202: never a
training (0-99999), validation (5042-5061) or test (1000-1900) seed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
NET = 'shaoxing_network_7'
MAP = os.path.join(FORK, 'maps', NET)
SIM_SEEDS = (900201, 900202)
N_VEHICLES = 6677


def _load_generator(source=None):
    """The build_file module: the working copy, or (source given) another version's text."""
    import importlib.util
    import types
    if source is None:
        spec = importlib.util.spec_from_file_location('fenglin_build', os.path.join(MAP, 'build_file.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    mod = types.ModuleType('fenglin_build_head')
    exec(compile(source, 'HEAD:build_file.py', 'exec'), mod.__dict__)
    return mod


def _cfg_text(mod, tmp, thread=7, seed=123):
    mod.gen_shaoxing_sumocfg_file(path=tmp, seed=seed, thread=thread)
    with open(tmp + '/shaoxing_network_7_{}.sumocfg'.format(thread)) as f:
        return f.read()


def _with_env(**kv):
    """Context manager: set / clear environment variables, restore them afterwards."""
    class Ctx:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kv}
            for k, v in kv.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = str(v)

        def __exit__(self, *exc):
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return Ctx()


def quick_checks():
    import shutil
    import tempfile
    results = []

    def record(name, ok, detail):
        results.append({'check': name, 'ok': bool(ok), 'detail': detail})
        print('[{}] {}: {}'.format('PASS' if ok else 'FAIL', name, detail), flush=True)

    head_src = subprocess.run(['git', 'show', 'HEAD:maps/shaoxing_network_7/build_file.py'], cwd=FORK,
                              capture_output=True, text=True).stdout
    head, cur = _load_generator(head_src), _load_generator()
    tmp = tempfile.mkdtemp(prefix='fenglin_cfg_')
    try:
        # the generator joins path + '/name', as MATSC_Env passes './maps/<net>' without a slash
        with _with_env(UNICORN_FCD='1', UNICORN_DEMAND=None):
            up, ours = _cfg_text(head, tmp), _cfg_text(cur, tmp)
        record('cfg_upstream', up == ours and 'fcd-output' in up,
               'UNICORN_FCD=1 sumocfg identical to HEAD\'s ({} bytes)'.format(len(up)))
        with _with_env(UNICORN_FCD=None, UNICORN_DEMAND=None):
            default = _cfg_text(cur, tmp)
        removed = [l for l in up.splitlines() if l not in default.splitlines()]
        added = [l for l in default.splitlines() if l not in up.splitlines()]
        record('cfg_default', not added and len(removed) == 2 and all('fcd-output' in l for l in removed),
               'default = upstream minus {} lines: {}'.format(len(removed), [l.strip() for l in removed]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # run_arlgos.ps1 restarted a RUNNING seed on 2026-11-16: under Windows PowerShell 5.1 a
    # function returning one CimInstance gives no .Count, so "(Running $s).Count -gt 0" was false.
    import re
    with open(os.path.join(HERE, 'run_arlgos.ps1'), encoding='utf-8') as f:
        launcher = f.read()
    bare = re.findall(r'(?<!@)\(Running \$s\)\.Count', launcher)
    wrapped = re.findall(r'@\(Running \$s\)\.Count', launcher)
    record('launcher_running_count', not bare and len(wrapped) >= 2,
           'run_arlgos.ps1 counts running seeds as @(Running $s).Count ({} places, {} unwrapped)'.format(
               len(wrapped), len(bare)))
    # 2026-11-22: under Windows PowerShell 5.1 a native command's stderr line, redirected with 2>&1
    # while $ErrorActionPreference is 'Stop', is a terminating error; every 2>&1 in the launcher must
    # run under 'Continue' (the exit code decides failure)
    lines = launcher.splitlines()
    redirects = [i for i, l in enumerate(lines) if '2>&1' in l and l.lstrip().startswith('& $Python')]
    guarded = [i for i in redirects if any("$ErrorActionPreference = 'Continue'" in l for l in lines[max(0, i - 3):i])]
    record('launcher_stderr_tolerant', redirects and len(guarded) == len(redirects),
           'run_arlgos.ps1: {} of {} python calls with 2>&1 run under ErrorActionPreference Continue'.format(
               len(guarded), len(redirects)))

    # 2026-11-22: a reader holding the queue log made Add-Content throw and the launcher die
    adds = [i for i, l in enumerate(lines) if 'Add-Content' in l and not l.lstrip().startswith('#')]
    safe = [i for i in adds if 'try' in lines[i] and any('catch' in l for l in lines[i:i + 2])]
    record('launcher_log_write_guarded', adds and len(safe) == len(adds),
           'run_arlgos.ps1: {} of {} Add-Content calls inside try/catch'.format(len(safe), len(adds)))

    base = cur.base_vehicles(MAP)
    by_id = {vid: (dep, edges) for vid, dep, edges in base}
    record('heldout_base', len(base) == N_VEHICLES and len(by_id) == N_VEHICLES,
           '{} vehicles with unique ids in {}'.format(len(base), cur.BASE_ROUTES))
    a1 = cur.heldout_vehicles(MAP, 5042, 60.0, 1.0)
    a2 = cur.heldout_vehicles(MAP, 5042, 60.0, 1.0)
    b = cur.heldout_vehicles(MAP, 1000, 60.0, 1.0)
    record('heldout_deterministic', a1 == a2, 'seed 5042 twice -> identical day')
    moved = sum(1 for x, y in zip(sorted(a1, key=lambda r: r[1]), sorted(b, key=lambda r: r[1])) if x[0] != y[0])
    record('heldout_seeds_differ', moved > 0.95 * N_VEHICLES,
           'seed 5042 vs 1000: {} of {} departure times differ'.format(moved, N_VEHICLES))
    times = [r[0] for r in a1]
    ok_routes = all(by_id[vid][1] == edges for _, _, vid, edges in a1)
    max_shift = max(abs(t - by_id[vid][0]) for t, _, vid, _ in a1)
    clamped = sum(1 for t, _, vid, _ in a1 if abs(t - by_id[vid][0]) > 60.0 + 1e-6)
    record('heldout_valid', len(a1) == N_VEHICLES and times == sorted(times) and min(times) >= 0.0
           and max(times) <= 3599.0 and ok_routes and clamped == 0,
           'keep=1: {} vehicles, sorted, {:.2f}-{:.2f} s, routes unchanged, max shift {:.2f} s, {} beyond jitter'
           .format(len(a1), min(times), max(times), max_shift, clamped))
    mean_abs = sum(abs(t - by_id[vid][0]) for t, _, vid, _ in a1) / len(a1)
    record('heldout_jitter_spread', 25.0 < mean_abs < 35.0,
           'mean |shift| {:.1f} s (uniform on [-60, 60] gives 30)'.format(mean_abs))
    thin = cur.heldout_vehicles(MAP, 5042, 60.0, 0.9)
    full = {vid: t for t, _, vid, _ in a1}
    same_times = all(full[vid] == t for t, _, vid, _ in thin)
    sd = (N_VEHICLES * 0.9 * 0.1) ** 0.5
    record('heldout_keep', abs(len(thin) - 0.9 * N_VEHICLES) < 4 * sd and same_times,
           'keep=0.9: {} of {} kept (expected {:.0f} +- {:.0f}); kept vehicles at the keep=1 times: {}'
           .format(len(thin), N_VEHICLES, 0.9 * N_VEHICLES, 4 * sd, same_times))
    return results


def episode(args):
    """One 3600 s episode with a seeded random phase sequence; prints one JSON line."""
    os.environ['UNICORN_NET'] = NET
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    sys.path.insert(0, HERE)
    from collections import OrderedDict
    import xml.etree.ElementTree as ET
    import numpy as np
    import pandas as pd
    from common import use_env_sumo
    use_env_sumo()
    from env.arlgos_env import ArlgosEnv
    from version_check import episode_metrics
    env = ArlgosEnv(server_number=args.server, test=True)
    if args.sim_seconds:
        env.max_test_step = args.sim_seconds
    from eval_unicorn import measure_at_observe
    measure_at_observe(env)               # eval_unicorn.py's validation score, recorded alongside
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    env.curr_episode = 0
    env.reset_seeded(args.seed)
    done, steps = False, 0
    while not done:
        act = OrderedDict((t, int(rng.randint(len(env.tls_dict[t].valid_phase_list)))) for t in env.rl_tls_list)
        _, _, done, _ = env.step(act)
        steps += 1
    seconds = time.time() - t0
    mirror = env.decision_queues
    mirror_diff = (max(abs(a - b) for a, b in zip(mirror, env.queue_series))
                   if len(mirror) == len(env.queue_series) else float('inf'))
    env.collect_trip_data()
    stats = ET.parse(env.statistic_output_file_path).getroot().find('vehicles').attrib
    traffic, trips = pd.DataFrame(env.traffic_data), pd.DataFrame(env.trip_data)

    def digest(df):
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    per_ep, _ = episode_metrics(traffic, trips)
    out = {'seed': args.seed, 'fcd': os.environ.get('UNICORN_FCD', '0'),
           'demand': os.environ.get('UNICORN_DEMAND', 'published') or 'published',
           'client': 'libsumo' if os.environ.get('UNICORN_LIBSUMO') == '1' else 'traci',
           'steps': steps, 'seconds': round(seconds, 1), 'traffic_rows': len(traffic), 'trips': len(trips),
           'traffic_sha': digest(traffic), 'trips_sha': digest(trips),
           'queue_sha': hashlib.sha256(json.dumps(env.queue_series).encode()).hexdigest(),
           'decision_queue': env.mean_queue(), 'loaded': int(stats['loaded']), 'inserted': int(stats['inserted']),
           'mirror_points': len(mirror), 'mirror_max_diff': mirror_diff,
           'metrics': {k: float(v) for k, v in per_ep[0].items()}}
    print('RESULT ' + json.dumps(out), flush=True)


def sim_checks(python, libsumo, sim_seconds=None):
    cases = [(s, fcd, 'published') for s in SIM_SEEDS for fcd in ('1', '0')] + [(SIM_SEEDS[0], '0', 'heldout')]
    procs = []
    for k, (seed, fcd, demand) in enumerate(cases):
        env = dict(os.environ, PYTHONPATH='', PYTHONIOENCODING='utf-8', UNICORN_FCD=fcd, UNICORN_DEMAND=demand,
                   UNICORN_LIBSUMO='1' if libsumo else '', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        cmd = [python, '-u', os.path.abspath(__file__), 'episode', '--seed', str(seed), '--server', str(420 + k)]
        if sim_seconds:
            cmd += ['--sim_seconds', str(sim_seconds)]
        procs.append(((seed, fcd, demand), subprocess.Popen(cmd, cwd=FORK, env=env, stdout=subprocess.PIPE,
                                                            stderr=subprocess.STDOUT, text=True)))
    runs = {}
    for case, p in procs:
        text = p.communicate()[0]
        lines = [l for l in text.splitlines() if l.startswith('RESULT ')]
        if p.returncode != 0 or not lines:
            raise SystemExit('episode {} failed:\n{}'.format(case, text[-3000:]))
        runs[case] = json.loads(lines[-1][len('RESULT '):])
        print('  episode {}: {} s, decision queue {:.4f}'.format(case, runs[case]['seconds'],
                                                                  runs[case]['decision_queue']), flush=True)
    results = []

    def record(name, ok, detail):
        results.append({'check': name, 'ok': bool(ok), 'detail': detail})
        print('[{}] {}: {}'.format('PASS' if ok else 'FAIL', name, detail), flush=True)
    keys = ('traffic_sha', 'trips_sha', 'queue_sha', 'loaded', 'inserted', 'traffic_rows', 'trips')
    for s in SIM_SEEDS:
        on, off = runs[(s, '1', 'published')], runs[(s, '0', 'published')]
        diff = [k for k in keys if on[k] != off[k]]
        record('fcd_equal_{}'.format(s), not diff,
               'fcd on vs off: {} traffic rows, {} trips, queue {:.4f}; differing: {}; {} s vs {} s'.format(
                   off['traffic_rows'], off['trips'], off['decision_queue'], diff or 'nothing',
                   on['seconds'], off['seconds']))
    worst = max(r['mirror_max_diff'] for r in runs.values())
    record('val_score_same', worst == 0.0,
           'eval_unicorn.decision_queue vs ArlgosEnv.queue_series at all {} decision points of {} episodes: '
           'max difference {}'.format(sum(r['mirror_points'] for r in runs.values()), len(runs), worst))
    pub, held = runs[(SIM_SEEDS[0], '0', 'published')], runs[(SIM_SEEDS[0], '0', 'heldout')]
    full_day = not sim_seconds
    record('heldout_sim', held['traffic_sha'] != pub['traffic_sha']
           and (not full_day or (held['loaded'] == pub['loaded'] == N_VEHICLES)),
           'published loaded {} / inserted {}; held-out loaded {} / inserted {}; traffic differs: {}'.format(
               pub['loaded'], pub['inserted'], held['loaded'], held['inserted'],
               held['traffic_sha'] != pub['traffic_sha']))
    return results, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='all', choices=['all', 'episode'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--server', type=int)
    ap.add_argument('--sim_seconds', type=int, default=None, help='shorter episodes (pipeline test only)')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--no_libsumo', action='store_true')
    ap.add_argument('--out', default=os.path.join(FORK, 'results', 'T15F'))
    args = ap.parse_args()
    if args.mode == 'episode':
        return episode(args)
    results = quick_checks()
    runs = {}
    if not args.quick:
        try:
            import libsumo  # noqa: F401
            libsumo_ok = not args.no_libsumo
        except ImportError:
            libsumo_ok = False
        sim, runs = sim_checks(args.python, libsumo_ok, args.sim_seconds)
        results += sim
    failed = [r for r in results if not r['ok']]
    print('\n{} of {} checks passed'.format(len(results) - len(failed), len(results)))
    if not args.quick and not args.sim_seconds:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'fenglin_check.json'), 'w') as f:
            json.dump({'checks': results, 'episodes': {'{}/fcd{}/{}'.format(*k): v for k, v in runs.items()}},
                      f, indent=1)
        md = ['# Fenglin switches: checks (T15F)', '', 'From `arlgos/fenglin_check.py`; {} of {} passed.'.format(
            len(results) - len(failed), len(results)), '', '| Check | Result | Detail |', '|---|---|---|']
        md += ['| {} | {} | {} |'.format(r['check'], 'pass' if r['ok'] else '**FAIL**', r['detail']) for r in results]
        md += ['', '| Episode (seed / fcd / demand) | Client | Seconds | Decision-point queue | Loaded | Inserted |',
               '|---|---|---|---|---|---|']
        md += ['| {} / {} / {} | {} | {} | {:.4f} | {} | {} |'.format(v['seed'], v['fcd'], v['demand'], v['client'],
                                                                     v['seconds'], v['decision_queue'], v['loaded'],
                                                                     v['inserted']) for v in runs.values()]
        with open(os.path.join(args.out, 'fenglin_check.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(md) + '\n')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
