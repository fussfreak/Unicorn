"""ARLGOS: training cost from a run_unicorn.ps1 log (T3 smoke run, T4 timing).

    python arlgos/timing.py --name smoke_cologne8 --out results/T3/smoke_timing

Reads logs/<name>.log (timestamped by run_logged.py), logs/<name>.mem.csv and
Train_MATSC/<name>/config.json, and writes <out>.md and <out>.json.

One update = the driver waits for one 3600 s episode from every worker, then
does the PPO update ("Finish updating the network"). The first update also
includes starting Ray and the workers, so the cost per update is the mean gap
between later updates. A run of MAX_EPISODES "episodes" makes MAX_EPISODES-1
updates (driver_unicorn.py starts its counter at 1).
"""
import argparse
import csv
import datetime
import json
import os
import re
import statistics

from common import REPO, machine_info, use_env_sumo

ANSI = re.compile(r'\x1b\[[0-9;]*m')
EPISODE = re.compile(r'\|\s*Reward:\s*(\S+),\s*Length:\s*(\d+),\s*Run Time:\s*([\d.]+)\s*s')


def parse_log(path):
    lines = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            stamp, _, text = raw.rstrip('\n').partition(' ')
            try:
                lines.append((datetime.datetime.fromisoformat(stamp), ANSI.sub('', text)))
            except ValueError:
                continue
    return lines


def read_config(run_dir):
    """Upstream writes config.json as 'key: value' lines, not JSON."""
    config = {}
    path = os.path.join(run_dir, 'config.json')
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                key, _, value = line.rstrip('\n').partition(': ')
                config[key] = value
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='run name (logs/<name>.log, Train_MATSC/<name>)')
    parser.add_argument('--out', required=True, help='output path without extension')
    parser.add_argument('--target', type=int, default=1500, help='MAX_EPISODES of a full run')
    args = parser.parse_args()

    lines = parse_log(os.path.join(REPO, 'logs', args.name + '.log'))
    start = lines[0][0]
    finishes, batches, batch = [], [], []
    for stamp, text in lines:
        match = EPISODE.search(text)
        if match:
            batch.append({'reward': float(match.group(1)), 'steps': int(match.group(2)),
                          'seconds': float(match.group(3))})
        elif 'Finish updating the network' in text:
            finishes.append(stamp)
            batches.append(batch)
            batch = []
    if len(finishes) < 2:
        raise SystemExit('need at least 2 finished updates in the log, found {}'.format(len(finishes)))

    gaps = [(b - a).total_seconds() for a, b in zip(finishes, finishes[1:])]
    first = (finishes[0] - start).total_seconds()
    per_update = statistics.mean(gaps)
    episode_seconds = [e['seconds'] for b in batches for e in b]
    slowest = [max(e['seconds'] for e in b) for b in batches[1:] if b]
    learner = per_update - statistics.mean(slowest) if slowest else None

    used = []
    mem_path = os.path.join(REPO, 'logs', args.name + '.mem.csv')
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            for row in csv.DictReader(f):
                if row['used_mb'] != 'used_mb':
                    used.append((int(row['used_mb']), int(row['total_mb'])))

    config = read_config(os.path.join(REPO, 'Train_MATSC', args.name))
    workers = int(config.get('NUM_META_AGENTS', len(batches[0]) or 0))
    updates_full = args.target - 1
    full_hours = per_update * updates_full / 3600
    use_env_sumo()
    result = {
        'run': args.name, 'net': config.get('NET_NAME'), 'workers': workers,
        'updates_timed': len(finishes), 'first_update_s': round(first, 1),
        'update_gaps_s': [round(g, 1) for g in gaps], 'seconds_per_update': round(per_update, 1),
        'episode_seconds': {'n': len(episode_seconds), 'mean': round(statistics.mean(episode_seconds), 1),
                            'max': round(max(episode_seconds), 1)},
        'learner_seconds_per_update': round(learner, 1) if learner is not None else None,
        'ram_mb': {'before': used[0][0] if used else None, 'peak': max(u for u, _ in used) if used else None,
                   'total': used[0][1] if used else None},
        'full_run': {'max_episodes': args.target, 'updates': updates_full,
                     'simulations': updates_full * workers, 'hours': round(full_hours, 1),
                     'days': round(full_hours / 24, 2)},
        'config': {k: config.get(k) for k in ('NET_NAME', 'GREEN_DURATION', 'YELLOW_DURATION', 'TELEPORT_TIME',
                                              'MAX_EPISODE', 'NUM_META_AGENTS', 'TRAIN_SEED', 'USE_GPU')},
        'machine': machine_info(),
    }
    with open(args.out + '.json', 'w') as f:
        json.dump(result, f, indent=2)

    m = result['machine']
    md = [
        '# Unicorn training cost: {} ({})'.format(args.name, result['net']),
        '',
        'Machine: {} ({} logical CPUs, {:.1f} GB RAM), Python {}, torch {}, Ray {}, SUMO {}. Fork commit {}.'.format(
            m['cpu'], m['logical_cpus'], m['ram_mb'] / 1024, m['python'], m['torch'], m['ray'], m['sumo'],
            m['fork_commit']),
        '',
        'One update = one 3600 s episode on each of the {} workers, then the PPO update.'.format(workers),
        '',
        '| | Value |',
        '|---|---|',
        '| Updates timed | {} (first one includes Ray start-up: {:.0f} s) |'.format(len(finishes), first),
        '| **Seconds per update** | **{:.0f}** (gaps: {}) |'.format(per_update, ', '.join('{:.0f}'.format(g) for g in gaps)),
        '| One worker episode (simulation + acting) | mean {:.0f} s, max {:.0f} s, n = {} |'.format(
            statistics.mean(episode_seconds), max(episode_seconds), len(episode_seconds)),
        '| Learner (PPO update + waiting) per update | {} |'.format(
            '{:.0f} s'.format(learner) if learner is not None else 'n/a'),
        '| RAM used (whole machine) | before {} MB, peak {} MB of {} MB |'.format(
            result['ram_mb']['before'], result['ram_mb']['peak'], result['ram_mb']['total']),
        '| **Full run ({} episodes = {} updates = {} simulations)** | **{:.0f} h = {:.1f} days** |'.format(
            args.target, updates_full, updates_full * workers, full_hours, full_hours / 24),
        '',
        'The full-run figure assumes every update costs what the timed ones did; it',
        'holds only for this machine and this number of workers.',
        '',
    ]
    with open(args.out + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
