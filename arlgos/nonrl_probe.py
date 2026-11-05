"""ARLGOS: headroom probe -- Unicorn's non-RL controllers on one network (SUMO 1.28 only).

    python arlgos/nonrl_probe.py summarize --root results/nonrl_probe/raw --out results/nonrl_probe/<net>

The runs themselves are `arlgos/version_check.py run` unchanged (launcher:
arlgos/nonrl_probe.ps1), so the controllers, seeds (1000, 1100, ..., 1900),
3600 s episodes, timings and metric definitions are exactly those of T3.

`summarize` reports, per controller, the paper's six metrics as the mean over
the 10 seeds with a 95 % CI (t, 9 df), and the pooled mean (std) the paper's
tables use. Next to them it prints the Unicorn and GESA rows the paper reports
for the same network, labelled as *reference only*: they come from the PDF, so
they say how much headroom there is above the non-RL controllers, never who wins
(ROADMAP fairness rule 1).
"""
import argparse
import json
import os

from common import machine_info
from version_check import AGENTS, METRICS, PAPER_NAME, load

# arXiv 2503.11488v1, Table IV (GESA datasets), Fenglin: mean (std), metric order as METRICS
PAPER_ROWS = {'shaoxing_network_7': {
    'GESA-single':     [(0.77, 0.28), (3.76, 1.60), (57.88, 21.72), (1.23, 0.66), (255.55, 391.00), (118.14, 289.99)],
    'GESA-multiple':   [(0.85, 0.34), (3.91, 1.58), (57.91, 22.95), (1.24, 0.64), (249.42, 381.99), (112.36, 281.71)],
    'Unicorn-single':  [(0.37, 0.08), (3.54, 1.54), (20.58, 6.32), (1.20, 0.64), (233.45, 417.07), (104.55, 322.79)],
    'Unicorn-multiple': [(0.37, 0.09), (3.56, 1.55), (20.74, 6.98), (1.20, 0.63), (232.14, 408.46), (102.72, 312.12)]}}


def summarize(args):
    from scipy import stats
    runs = load(args.root)
    nets = sorted({k[1] for k in runs})
    md, out = ['# Non-RL headroom probe', ''], {'runs': {}, 'machine': machine_info()}
    md += ['Unicorn\'s own non-RL controllers (`evaluator_non_rl.py` via `arlgos/version_check.py run`,',
           'unchanged) on its 10 test seeds (1000, 1100, ..., 1900), 3600 s each, SUMO {}.'.format(
               ', '.join(sorted({k[0] for k in runs}))), '']
    for net in nets:
        md += ['## {}'.format(net), '',
               'Mean over the 10 seeds [95 % CI], then the pooled mean (std) over all steps or trips, as in',
               'the paper\'s tables.', '',
               '| Controller | ' + ' | '.join(label for _, label, _ in METRICS) + ' |',
               '|---|' + '---|' * len(METRICS)]
        for version, net_, agent in sorted(runs):
            if net_ != net:
                continue
            per_ep, pooled, done = runs[(version, net_, agent)]
            cells, rec = [], {}
            for key, _, _ in METRICS:
                x = [per_ep[e][key] for e in sorted(per_ep)]
                m = sum(x) / len(x)
                half = stats.t.ppf(0.975, len(x) - 1) * stats.tstd(x) / len(x) ** 0.5
                cells.append('{:.2f} [{:.2f}, {:.2f}]<br>{:.2f} ({:.2f})'.format(
                    m, m - half, m + half, pooled[key][0], pooled[key][1]))
                rec[key] = {'seed_mean': m, 'ci95': half, 'pooled': list(pooled[key]), 'per_seed': x}
            md.append('| {} | {} |'.format(PAPER_NAME[agent], ' | '.join(cells)))
            out['runs']['{}/{}/{}'.format(version, net, agent)] = {
                'metrics': rec, 'timing': done['timing'], 'sumo_reported': done['sumo_reported'],
                'episode_seconds': done['episode_seconds']}
        if net in PAPER_ROWS:
            md += ['', '**Reference only (from the PDF, not rerun):** the paper\'s learned rows for this '
                   'network, pooled mean (std).', '',
                   '| Paper row | ' + ' | '.join(label for _, label, _ in METRICS) + ' |',
                   '|---|' + '---|' * len(METRICS)]
            for name, vals in PAPER_ROWS[net].items():
                md.append('| {} | {} |'.format(name, ' | '.join('{:.2f} ({:.2f})'.format(m, s) for m, s in vals)))
        md.append('')
    md += ['## Runs', '', '| Network | Controller | SUMO reported | Green/yellow/teleport | s per episode |',
           '|---|---|---|---|---|']
    for (version, net, agent), (_, _, done) in sorted(runs.items()):
        md.append('| {} | {} | {} | {} | {:.0f} |'.format(
            net, agent, ', '.join(done['sumo_reported']), '/'.join(str(t) for t in done['timing']),
            sum(done['episode_seconds']) / len(done['episode_seconds'])))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    with open(args.out + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_sum = sub.add_parser('summarize')
    p_sum.add_argument('--root', required=True)
    p_sum.add_argument('--out', required=True, help='output path without extension')
    args = parser.parse_args()
    summarize(args)


if __name__ == '__main__':
    main()
