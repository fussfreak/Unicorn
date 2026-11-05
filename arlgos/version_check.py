"""ARLGOS: SUMO version check with Unicorn's own non-RL baselines (T3).

    python arlgos/version_check.py run --net cologne_network_8 --agent FIXED --server 70 \
        --out results/T3/raw/sumo-1.28.0/cologne_network_8
    python arlgos/version_check.py summarize --root results/T3/raw --out results/T3/version_check

`run` is evaluator_non_rl.py's main loop for one network and one agent:
upstream's Evaluator class unchanged, the same 10 seeds (1000, 1100, ..., 1900),
the same per-step and per-trip CSVs, plus a <AGENT>.done.json with the SUMO
version that actually ran. Its own --server number gives each parallel run
its own TraCI port and its own files next to the map.

`summarize` turns those CSVs into the paper's six metrics:
  queue length   mean over steps of avg_queue (halting vehicles per incoming lane)
  speed          mean over steps of avg_speed_mps (all vehicles in the network)
  delay          mean over steps of avg_wait_sec ("intersection delay")
  completion     mean over steps of vehicles arriving per step (veh/s)
  trip time      mean duration_sec of finished trips
  trip delay     mean time_loss_sec of finished trips
and compares SUMO 1.16 with 1.28 seed by seed, and Grid 5x5 with Table II.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

from common import REPO, machine_info

AGENTS = ('FIXED', 'GREEDY', 'PRESSURE')
METRICS = [('queue', 'Queue length (veh)', 'lower'), ('speed', 'Speed (m/s)', 'higher'),
           ('delay', 'Intersection delay (s)', 'lower'), ('completion', 'Trip completion (veh/s)', 'higher'),
           ('trip_time', 'Trip time (s)', 'lower'), ('trip_delay', 'Trip delay (s)', 'lower')]
# Unicorn paper, Table II, Grid 5x5: mean (std); same metric order as METRICS
TABLE_II_GRID = {'FIXED': [(3.16, 1.57), (2.03, 1.56), (29.34, 17.65), (0.62, 0.54), (657.67, 551.72), (411.95, 428.68)],
                 'GREEDY': [(2.65, 1.72), (3.27, 2.42), (42.09, 38.05), (0.86, 0.41), (546.96, 429.35), (263.59, 325.64)],
                 'PRESSURE': [(2.79, 1.76), (2.93, 2.19), (39.17, 37.50), (0.82, 0.42), (572.43, 404.92), (270.41, 292.18)]}
PAPER_NAME = {'FIXED': 'Fixed-Time', 'GREEDY': 'Greedy', 'PRESSURE': 'Max-Pressure'}


def run(args):
    os.environ['UNICORN_NET'] = args.net          # parameters.py reads it at import
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import numpy as np
    import torch
    import traci
    import evaluator_non_rl
    from parameters import SUMO_PARAMS
    from utils import set_env
    evaluator_non_rl.os = os                      # upstream imports os only under __main__

    random.seed(21)                               # evaluator_non_rl.py __main__
    np.random.seed(21)
    torch.manual_seed(21)
    seeds = SUMO_PARAMS.TEST_SEEDS                # = start_seed + seed * i, i < test_num

    env = set_env(server_number=args.server, test=True)
    reported = []
    set_world = env.set_world

    def checked_set_world(seed, write_unfinished=False):
        if set_world(seed, write_unfinished) != 0:
            raise RuntimeError('SUMO did not start (seed {}, server {})'.format(seed, args.server))
        reported.append(traci.getVersion())
        return 0

    env.set_world = checked_set_world
    # the three agents share args.out and start together; Evaluator's exists-then-makedirs races
    os.makedirs(os.path.join(args.out, 'eval_data'), exist_ok=True)
    evaluator = evaluator_non_rl.Evaluator(env=env, exp_dir=args.out, agent_name=args.agent)
    seconds = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        evaluator.evaluate(index=i, seed=seed)
        seconds.append(round(time.time() - t0, 1))
    evaluator.output_eval_data()

    done = {'net': args.net, 'agent': args.agent, 'seeds': seeds, 'server': args.server,
            'timing': [SUMO_PARAMS.GREEN_DURATION, SUMO_PARAMS.YELLOW_DURATION, SUMO_PARAMS.TELEPORT_TIME],
            'sumo_reported': sorted({v[1] for v in reported}), 'traci_api_server': sorted({v[0] for v in reported}),
            'episode_seconds': seconds, 'machine': machine_info()}
    with open(os.path.join(args.out, '{}.done.json'.format(args.agent)), 'w') as f:
        json.dump(done, f, indent=2)


def episode_metrics(traffic, trips):
    """Per-episode (= per-seed) values of the six metrics, plus the pooled mean(std) the paper reports."""
    per_episode = {}
    for episode, steps in traffic.groupby('episode'):
        finished = trips[trips['episode'] == episode]
        per_episode[int(episode)] = {'queue': steps['avg_queue'].mean(), 'speed': steps['avg_speed_mps'].mean(),
                                     'delay': steps['avg_wait_sec'].mean(),
                                     'completion': steps['number_arrived_car'].mean(),
                                     'trip_time': finished['duration_sec'].astype(float).mean(),
                                     'trip_delay': finished['time_loss_sec'].astype(float).mean()}
    pooled = {'queue': traffic['avg_queue'], 'speed': traffic['avg_speed_mps'], 'delay': traffic['avg_wait_sec'],
              'completion': traffic['number_arrived_car'], 'trip_time': trips['duration_sec'].astype(float),
              'trip_delay': trips['time_loss_sec'].astype(float)}
    return per_episode, {k: (v.mean(), v.std()) for k, v in pooled.items()}


def load(root):
    """{(version, net, agent): (per_episode, pooled, done)} from every finished run under root."""
    import pandas as pd
    runs = {}
    for done_path in glob.glob(os.path.join(root, 'sumo-*', '*', '*.done.json')):
        with open(done_path) as f:
            done = json.load(f)
        version = os.path.basename(os.path.dirname(os.path.dirname(done_path)))[len('sumo-'):]
        prefix = '_'.join(done['net'].split('_')[:2])          # Evaluator.output_eval_data's file names
        data = os.path.join(os.path.dirname(done_path), 'eval_data', prefix + '_' + done['agent'])
        traffic, trips = pd.read_csv(data + '_traffic.csv'), pd.read_csv(data + '_trip.csv')
        runs[(version, done['net'], done['agent'])] = episode_metrics(traffic, trips) + (done,)
    return runs


def summarize(args):
    import warnings
    from scipy import stats
    warnings.filterwarnings('ignore', message='(Exact p-value|Sample size too small)')   # ties in wilcoxon
    runs = load(args.root)
    versions = sorted({k[0] for k in runs}, key=lambda v: [int(x) for x in v.split('.')])
    nets = sorted({k[1] for k in runs})
    md = ['# SUMO version check (T3)', '',
          'Unicorn\'s own non-RL controllers (`evaluator_non_rl.py`, unchanged) on its 10 test seeds',
          '(1000, 1100, ..., 1900), 3600 s each, with SUMO {}. Each run recorded the version SUMO'.format(
              ' and '.join(versions)),
          'itself reported over TraCI. No training is involved, so any difference comes from the simulator.', '']
    summary = {'versions': versions, 'runs': {}}
    for (version, net, agent), (_, pooled, done) in sorted(runs.items()):
        summary['runs']['{}/{}/{}'.format(version, net, agent)] = {
            'sumo_reported': done['sumo_reported'], 'timing': done['timing'],
            'pooled': {k: [round(m, 4), round(s, 4)] for k, (m, s) in pooled.items()},
            'episode_seconds': done['episode_seconds']}
        if not any(version in v for v in done['sumo_reported']):
            md.append('**WARNING:** {}/{}/{} asked for SUMO {} but SUMO reported {}.'.format(
                version, net, agent, version, done['sumo_reported']))

    if len(versions) == 2:
        old, new = versions
        md += ['## 1. Simulator effect: SUMO {} minus {}, paired by seed'.format(new, old), '',
               'Mean of the per-seed differences, with a 95 % CI (t, 9 df) and a two-sided Wilcoxon',
               'signed-rank p (descriptive, not corrected for the many cells). Relative = difference / {} mean.'.format(old), '']
        for net in nets:
            md += ['### {}'.format(net), '',
                   '| Agent | Metric | SUMO {} | SUMO {} | Difference [95 % CI] | Relative | p |'.format(old, new),
                   '|---|---|---|---|---|---|---|']
            for agent in AGENTS:
                if (old, net, agent) not in runs or (new, net, agent) not in runs:
                    continue
                a, b = runs[(old, net, agent)][0], runs[(new, net, agent)][0]
                episodes = sorted(set(a) & set(b))
                for key, label, _ in METRICS:
                    x = [a[e][key] for e in episodes]
                    y = [b[e][key] for e in episodes]
                    diff = [j - i for i, j in zip(x, y)]
                    mean_x, mean_d = sum(x) / len(x), sum(diff) / len(diff)
                    half = stats.t.ppf(0.975, len(diff) - 1) * stats.tstd(diff) / len(diff) ** 0.5 if len(diff) > 1 else float('nan')
                    p = stats.wilcoxon(diff).pvalue if any(diff) else 1.0
                    md.append('| {} | {} | {:.3f} | {:.3f} | {:+.3f} [{:+.3f}, {:+.3f}] | {:+.1f} % | {:.3f} |'.format(
                        PAPER_NAME[agent], label, mean_x, sum(y) / len(y), mean_d, mean_d - half, mean_d + half,
                        100 * mean_d / mean_x if mean_x else float('nan'), p))
                    summary.setdefault('differences', {})['{}/{}/{}'.format(net, agent, key)] = {
                        'old': mean_x, 'new': sum(y) / len(y), 'diff': mean_d, 'ci95': half, 'p': p}
            md.append('')

    if any(k[1] == 'grid_network_5_5' for k in runs):
        md += ['## 2. Setup check: Grid 5x5 against the paper\'s Table II', '',
               'Pooled mean (std) over all steps (or all finished trips) of the 10 episodes, the way',
               'Table II reports them. "Within" = our mean lies within one paper std of the paper\'s mean.', '',
               '| Agent | Metric | Paper | ' + ' | '.join('SUMO {}'.format(v) for v in versions) + ' |',
               '|---|---|---|' + '---|' * len(versions)]
        for agent in AGENTS:
            for i, (key, label, _) in enumerate(METRICS):
                paper_mean, paper_std = TABLE_II_GRID[agent][i]
                cells = []
                for version in versions:
                    run_ = runs.get((version, 'grid_network_5_5', agent))
                    if run_ is None:
                        cells.append('n/a')
                        continue
                    mean, std = run_[1][key]
                    within = 'within' if abs(mean - paper_mean) <= paper_std else 'outside'
                    cells.append('{:.2f} ({:.2f}), {}'.format(mean, std, within))
                md.append('| {} | {} | {:.2f} ({:.2f}) | {} |'.format(
                    PAPER_NAME[agent], label, paper_mean, paper_std, ' | '.join(cells)))
        md.append('')

    md += ['## Runs', '', '| SUMO asked | Network | Agent | SUMO reported | Green/yellow/teleport | s per episode |',
           '|---|---|---|---|---|---|']
    for (version, net, agent), (_, _, done) in sorted(runs.items()):
        md.append('| {} | {} | {} | {} | {} | {:.0f} |'.format(
            version, net, agent, ', '.join(done['sumo_reported']), '/'.join(str(t) for t in done['timing']),
            sum(done['episode_seconds']) / len(done['episode_seconds'])))
    md.append('')
    summary['machine'] = machine_info()
    with open(args.out + '.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    with open(args.out + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_run = sub.add_parser('run')
    p_run.add_argument('--net', required=True)
    p_run.add_argument('--agent', required=True, choices=AGENTS)
    p_run.add_argument('--server', type=int, required=True, help='TraCI port 6000+server; unique per parallel run')
    p_run.add_argument('--out', required=True)
    p_sum = sub.add_parser('summarize')
    p_sum.add_argument('--root', required=True)
    p_sum.add_argument('--out', required=True, help='output path without extension')
    args = parser.parse_args()
    if args.cmd == 'run':
        args.out = os.path.abspath(args.out)
        run(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
