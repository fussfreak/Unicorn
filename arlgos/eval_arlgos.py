"""ARLGOS (T14): test a trained ARLGOS-v2 run the way Unicorn's evaluator_rl.py tests Unicorn.

    python arlgos/eval_arlgos.py --run Train_MATSC/arlgos_cologne_network_8_seed1
    python arlgos/eval_arlgos.py --summary cologne_network_8        # all tested seeds -> one table
    python arlgos/eval_arlgos.py --run ... --watch                  # one episode in sumo-gui

Test = the checkpoint chosen on validation (model/best_val.pt), played greedily, once, on
Unicorn's 10 test seeds (1000, 1100, ..., 1900), 3600 s each. The measurements are the
environment's own, unchanged (MATSC_Env.measure_traffic_step every simulated second and
collect_trip_data), written in evaluator_rl.py's two CSV formats, and turned into the paper's
six metrics by arlgos/version_check.py's episode_metrics, the same code that scored the
fixed-time, greedy and max-pressure rows in T3. So an ARLGOS row and a baseline row are
computed by one function from one kind of file.

Greedy means argmax_p Q_j(p) over each junction's real phases. No random number is drawn, so
the same checkpoint and seed always give the same episode.

Output: <run>/eval/{eval_data/*.csv, test.json, test.md, config.json}. A finished test is never
overwritten (test seeds are used once per configuration); --force exists for a changed
evaluator, and says so in config.json.
--watch plays demo seed 9042 in sumo-gui and writes nothing (needs the traci client, i.e.
UNICORN_LIBSUMO unset).
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WATCH_SEED = 9042
METRIC_KEYS = ('queue', 'speed', 'delay', 'completion', 'trip_time', 'trip_delay')


def load_run_config(run):
    with open(os.path.join(run, 'config.json')) as f:
        return json.load(f)


def evaluate(args):
    run = os.path.abspath(os.path.join(REPO, args.run) if not os.path.isabs(args.run) else args.run)
    cfg = load_run_config(run)
    out = os.path.join(run, 'eval')
    if os.path.exists(os.path.join(out, 'test.json')) and not (args.force or args.watch):
        raise SystemExit('{} is already tested (test seeds are used once); --force to redo'.format(run))
    demand = os.environ.get('UNICORN_DEMAND', 'published') or 'published'
    if cfg.get('demand', 'published') != demand:          # T15F: test on the demand the run trained on
        raise SystemExit('the run trained with UNICORN_DEMAND={} but this shell has {}'.format(
            cfg.get('demand', 'published'), demand))
    os.environ['UNICORN_NET'] = cfg['net']
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    import numpy as np
    import pandas as pd
    import torch
    from parameters import SUMO_PARAMS
    from common import machine_info, use_env_sumo
    from env.arlgos_env import ArlgosEnv
    from models.Arlgos import ArlgosAgent
    from version_check import METRICS, episode_metrics
    use_env_sumo()
    torch.set_num_threads(1)

    server = 0 if args.watch else (args.server if args.server is not None else 300 + int(cfg['seed']))
    env = ArlgosEnv.from_run_config(cfg, server, test=True)     # same reward and lane_weight as training
    if args.watch:
        if os.environ.get('UNICORN_LIBSUMO') == '1':
            raise SystemExit('--watch needs the traci client: unset UNICORN_LIBSUMO')
        env.gui = True
    agent = ArlgosAgent(env.graph_spec(), episodes=1, seed=0, algo=cfg['algo'], conv=cfg['conv'],
                        inter=not cfg['no_inter'], agent_id=not cfg['no_agent_id'], hidden=cfg['hidden'],
                        mixer_embed=cfg['mixer_embed'])
    ckpt_path = os.path.join(run, 'model', args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    agent.load_policy(ckpt)
    seeds = [WATCH_SEED] if args.watch else list(SUMO_PARAMS.TEST_SEEDS)
    if args.sim_seconds or args.n_seeds:           # pipeline checks only; recorded in config.json
        env.max_test_step = args.sim_seconds or env.max_test_step
        seeds = seeds[:args.n_seeds or len(seeds)]
    seconds = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        env.curr_episode = i
        obs = env.flat_obs(env.reset_seeded(seed))
        cur, done, steps, switches = env.current_phases(), False, 0, 0.0
        while not done:
            actions = agent.greedy(obs, cur)
            nxt, _, done, info = env.step(OrderedDict((t, int(actions[j])) for j, t in enumerate(env.rl_tls_list)))
            obs, cur = env.flat_obs(nxt), env.current_phases()
            steps += 1
            switches += info[1]
        if not args.watch:
            env.collect_trip_data()
        seconds.append(round(time.time() - t0, 1))
        print('[TEST] seed {}  {} steps  decision-point queue {:.3f}  switch rate {:.2f}  {:.0f} s'.format(
            seed, steps, env.mean_queue(), switches / steps, seconds[-1]), flush=True)
    if args.watch:
        return

    data = os.path.join(out, 'eval_data')
    os.makedirs(data, exist_ok=True)
    prefix = '_'.join(cfg['net'].split('_')[:2])              # evaluator_rl.py's file names
    traffic, trips = pd.DataFrame(env.traffic_data), pd.DataFrame(env.trip_data)
    traffic.to_csv(os.path.join(data, prefix + '_ARLGOS_traffic.csv'))
    trips.to_csv(os.path.join(data, prefix + '_ARLGOS_trip.csv'))
    per_episode, pooled = episode_metrics(traffic, trips)
    result = {'run': os.path.basename(run), 'net': cfg['net'], 'method': 'ARLGOS-v2', 'mode': 'greedy',
              'demand': demand, 'checkpoint': args.checkpoint,
              'checkpoint_episode': ckpt.get('episode'), 'val_queue': ckpt.get('val_queue'),
              'seeds': seeds,
              'per_seed': {k: [float(per_episode[e][k]) for e in sorted(per_episode)] for k in METRIC_KEYS},
              'pooled': {k: [float(pooled[k][0]), float(pooled[k][1])] for k in METRIC_KEYS},
              'episode_seconds': seconds}
    with open(os.path.join(out, 'test.json'), 'w') as f:
        json.dump(result, f, indent=1)
    md = ['# Test: {}'.format(result['run']), '',
          '`{}` (episode {}, validation queue {}) on seeds {}, greedy, {} s episodes.'.format(
              args.checkpoint, result['checkpoint_episode'], result['val_queue'], seeds, env.max_test_step), '',
          '| Metric | Mean over seeds | Pooled mean (std), as the paper reports | Better |', '|---|---|---|---|']
    for key, label, better in METRICS:
        x = result['per_seed'][key]
        md.append('| {} | {:.3f} | {:.2f} ({:.2f}) | {} |'.format(label, sum(x) / len(x), pooled[key][0],
                                                              pooled[key][1], better))
    with open(os.path.join(out, 'test.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({'train_config': cfg, 'checkpoint': args.checkpoint, 'seeds': seeds, 'forced': bool(args.force),
                   'pipeline_check': bool(args.sim_seconds or args.n_seeds), 'episode_s': env.max_test_step,
                   'demand': demand, 'fcd': os.environ.get('UNICORN_FCD', '0') or '0',
                   'lane_weight': env.lane_weight,
                   'sumo_client': 'libsumo' if os.environ.get('UNICORN_LIBSUMO') == '1' else 'traci',
                   'machine': machine_info()}, f, indent=1, default=str)
    print('\n'.join(md))


def summary(args):
    """results/arlgos_v2/<net>.md: every tested ARLGOS-v2 seed of one configuration prefix."""
    runs = sorted(glob.glob(os.path.join(REPO, 'Train_MATSC', '{}_{}_seed*'.format(args.prefix, args.summary),
                                         'eval', 'test.json')))
    if not runs:
        raise SystemExit('no tested runs named {}_{}_seed*'.format(args.prefix, args.summary))
    results = []
    for path in runs:
        with open(path) as f:
            results.append(json.load(f))
    sys.path.insert(0, HERE)
    from version_check import METRICS
    md = ['# ARLGOS-v2 on {} ({} training seeds)'.format(args.summary, len(results)), '',
          'Each cell: mean over the 10 test seeds of that run\'s chosen checkpoint. Last column: mean over the',
          'training seeds. Baseline rows and the Unicorn rerun are added by the head-to-head report (T15/T17).', '',
          '| Metric | ' + ' | '.join(r['run'].split('_')[-1] for r in results) + ' | Mean |',
          '|---|' + '---|' * (len(results) + 1)]
    for key, label, _ in METRICS:
        cells = [sum(r['per_seed'][key]) / len(r['per_seed'][key]) for r in results]
        md.append('| {} | {} | {:.3f} |'.format(label, ' | '.join('{:.3f}'.format(c) for c in cells),
                                              sum(cells) / len(cells)))
    md += ['', 'Chosen checkpoints: ' + ', '.join('{} ep {} (val {:.4f})'.format(
        r['run'].split('_')[-1], r['checkpoint_episode'], r['val_queue'] or float('nan')) for r in results)]
    out = os.path.join(REPO, 'results', 'arlgos_v2')
    os.makedirs(out, exist_ok=True)
    name = '{}_{}.md'.format(args.prefix, args.summary)
    with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('\nwrote', os.path.join(out, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='Train_MATSC/<name>')
    ap.add_argument('--checkpoint', default='best_val.pt')
    ap.add_argument('--server', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--sim_seconds', type=int, default=None, help='shorter test episodes: pipeline checks only')
    ap.add_argument('--n_seeds', type=int, default=None, help='only the first n test seeds: pipeline checks only')
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--summary', metavar='NET', help='aggregate the tested seeds of this network')
    ap.add_argument('--prefix', default='arlgos', help='run-name prefix for --summary')
    args = ap.parse_args()
    if args.summary:
        summary(args)
    elif args.run:
        evaluate(args)
    else:
        ap.error('give --run or --summary')


if __name__ == '__main__':
    main()
