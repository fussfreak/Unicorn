"""ARLGOS (T15F): the Fenglin head-to-head table, with the scoring fixed before any result.

    python arlgos/fenglin_report.py                     # published protocol
    python arlgos/fenglin_report.py --demand heldout    # held-out traffic days

1. Collects every tested run into results/T15F/runs/<run>/ (test.json, test_sampled.json,
   best.json / val.csv, config.json). That folder is small and is what travels through git: the
   lab PC pushes its Unicorn runs there, the laptop its ARLGOS-v2 runs, and the table is built
   from it on either machine.
2. Writes results/T15F/fenglin_h2h[_heldout].{md,json}.

Pre-registered scoring (decided 2026-11-16, before any Fenglin result; docs/T15F.md section 2):
  * metrics: the paper's six, from arlgos/version_check.episode_metrics, on Unicorn's 10 test
    seeds (1000, 1100, ..., 1900), 3600 s each; the same function scored every row;
  * per test seed, a method's value = the mean over its 3 training seeds;
  * primary: queue length. Two-sided Wilcoxon signed-rank over the 10 paired test seeds;
    "match" = ARLGOS-v2 within 10 % of Unicorn (ARLGOS <= 1.10 x Unicorn);
    "beats" = ARLGOS-v2 lower with p < 0.05;
  * secondary: intersection delay and trip delay, Wilcoxon, Holm-corrected over the two;
  * speed, completion and trip time: reported, not tested;
  * both methods greedy (argmax), checkpoints chosen on validation seeds 5042-5046.
    Unicorn's sampled evaluation (how evaluator_rl.py scores it) is shown as an extra row.
The non-RL rows come from results/nonrl_probe/shaoxing_network_7.json (published day only).
"""
import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
NET = 'shaoxing_network_7'
OUT = os.path.join(FORK, 'results', 'T15F')
KEYS = ('queue', 'speed', 'delay', 'completion', 'trip_time', 'trip_delay')
LABEL = {'queue': 'Queue (veh)', 'speed': 'Speed (m/s)', 'delay': 'Int. delay (s)',
         'completion': 'Completion (veh/s)', 'trip_time': 'Trip time (s)', 'trip_delay': 'Trip delay (s)'}
LOWER = {'queue', 'delay', 'trip_time', 'trip_delay'}
PRIMARY, SECONDARY = 'queue', ('delay', 'trip_delay')
MATCH_MARGIN = 0.10
# arXiv 2503.11488v1, Table IV, Fenglin, pooled means: REFERENCE ONLY (fairness rule 1)
PAPER_UNICORN = {'queue': 0.37, 'speed': 3.54, 'delay': 20.58, 'completion': 1.20, 'trip_time': 233.45,
                 'trip_delay': 104.55}
COPY = (('eval/test.json', 'test.json'), ('eval/test_sampled.json', 'test_sampled.json'),
        ('eval/best.json', 'best.json'), ('eval/val.csv', 'val_unicorn.csv'), ('val.csv', 'val.csv'),
        ('config.json', 'config.json'), ('eval/config.json', 'eval_config.json'), ('done.json', 'done.json'))


PREFIX_OVERRIDE = {}


def prefixes(demand):
    """Run-name prefixes per method; --arlgos_prefix / --unicorn_prefix override them (e.g. a
    lab-PC rerun of ARLGOS-v2, so that one table holds runs from one machine)."""
    p = {'ARLGOS-v2': 'arlgos_heldout' if demand == 'heldout' else 'arlgos',
         'Unicorn': 'unicorn_heldout' if demand == 'heldout' else 'unicorn'}
    p.update(PREFIX_OVERRIDE)
    return p


def collect(demand):
    """Copy the small result files of every tested run into results/T15F/runs/<run>/."""
    copied = []
    for method, prefix in prefixes(demand).items():
        for run in sorted(glob.glob(os.path.join(FORK, 'Train_MATSC', '{}_{}_seed*'.format(prefix, NET)))):
            if not os.path.exists(os.path.join(run, 'eval', 'test.json')):
                continue
            dst = os.path.join(OUT, 'runs', os.path.basename(run))
            os.makedirs(dst, exist_ok=True)
            for src, name in COPY:
                s = os.path.join(run, src)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(dst, name))
            copied.append(os.path.basename(run))
    return copied


def load(demand):
    runs = {m: {} for m in prefixes(demand)}
    sampled = {}
    for method, prefix in prefixes(demand).items():
        for d in sorted(glob.glob(os.path.join(OUT, 'runs', '{}_{}_seed*'.format(prefix, NET)))):
            with open(os.path.join(d, 'test.json')) as f:
                t = json.load(f)
            runs[method][os.path.basename(d)] = t
            sp = os.path.join(d, 'test_sampled.json')
            if os.path.exists(sp):
                with open(sp) as f:
                    sampled[os.path.basename(d)] = json.load(f)
    return runs, sampled


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demand', default='published', choices=['published', 'heldout'])
    ap.add_argument('--no_collect', action='store_true')
    ap.add_argument('--arlgos_prefix', default=None)
    ap.add_argument('--unicorn_prefix', default=None)
    ap.add_argument('--tag', default='', help='suffix for the output file names')
    args = ap.parse_args()
    if args.arlgos_prefix:
        PREFIX_OVERRIDE['ARLGOS-v2'] = args.arlgos_prefix
    if args.unicorn_prefix:
        PREFIX_OVERRIDE['Unicorn'] = args.unicorn_prefix
    import numpy as np
    from scipy.stats import wilcoxon
    copied = [] if args.no_collect else collect(args.demand)
    runs, sampled = load(args.demand)
    seed_lists = {tuple(r['seeds']) for group in list(runs.values()) + [sampled] for r in group.values()}
    if len(seed_lists) > 1:                              # pairing by position needs one seed list
        raise SystemExit('runs were tested on different seed lists: {}'.format(sorted(seed_lists)))

    def per_seed(group, key):
        """(10,) mean over the group's training seeds of each test seed's value."""
        return np.mean(np.array([r['per_seed'][key] for r in group.values()], dtype=float), axis=0)

    rows = []                                            # (label, n runs, {key: mean})
    for method in ('ARLGOS-v2', 'Unicorn'):
        if runs[method]:
            rows.append(('**{}** (greedy)'.format(method), len(runs[method]),
                         {k: float(per_seed(runs[method], k).mean()) for k in KEYS}))
    uni_sampled = {k: v for k, v in sampled.items() if k in runs['Unicorn']}
    if uni_sampled:
        rows.append(('Unicorn (sampled, as evaluator_rl.py)', len(uni_sampled),
                     {k: float(per_seed(uni_sampled, k).mean()) for k in KEYS}))
    probe = os.path.join(FORK, 'results', 'nonrl_probe', '{}.json'.format(NET))
    if args.demand == 'published' and os.path.exists(probe):
        with open(probe) as f:
            p = json.load(f)['runs']
        for agent, label in (('FIXED', 'Fixed-time'), ('GREEDY', 'Greedy'), ('PRESSURE', 'Max-pressure')):
            rec = p.get('1.28.0/{}/{}'.format(NET, agent))
            if rec:
                rows.append((label, 1, {k: float(rec['metrics'][k]['seed_mean']) for k in KEYS}))

    title = 'held-out traffic days' if args.demand == 'heldout' else 'published protocol (fixed route file)'
    md = ['# T15F: ARLGOS-v2 vs Unicorn on Fenglin ({})'.format(title), '',
          'Unicorn\'s 10 test seeds (1000, 1100, ..., 1900), 3600 s each, SUMO 1.28.0. Each cell: mean over',
          'the test seeds of the mean over a method\'s training seeds. Every row is scored by the same',
          'function (`arlgos/version_check.episode_metrics`). Lower is better except speed and completion.', '',
          '| Method | Runs | ' + ' | '.join(LABEL[k] for k in KEYS) + ' |', '|---|---|' + '---|' * len(KEYS)]
    for label, n, vals in rows:
        md.append('| {} | {} | {} |'.format(label, n, ' | '.join('{:.3f}'.format(vals[k]) for k in KEYS)))
    if args.demand == 'published':
        md.append('| *Unicorn-single, paper Table IV (PDF, reference only)* | - | {} |'.format(
            ' | '.join('*{:.2f}*'.format(PAPER_UNICORN[k]) for k in KEYS)))
    md += ['', '## Runs', '', '| Run | Chosen checkpoint | Validation queue | Test queue (greedy) |',
           '|---|---|---|---|']
    for method in ('ARLGOS-v2', 'Unicorn'):
        for name, r in runs[method].items():
            ck = r.get('checkpoint_episode', r.get('checkpoint'))
            md.append('| {} | {} | {} | {:.3f} |'.format(name, ck, '{:.4f}'.format(r['val_queue'])
                                                         if r.get('val_queue') is not None else '-',
                                                         float(np.mean(r['per_seed']['queue']))))
    verdict = None
    if runs['ARLGOS-v2'] and runs['Unicorn']:
        a, u = per_seed(runs['ARLGOS-v2'], PRIMARY), per_seed(runs['Unicorn'], PRIMARY)
        p = float(wilcoxon(a, u).pvalue) if np.any(a != u) else 1.0
        diff = a - u
        rel = (a.mean() - u.mean()) / u.mean()
        match = a.mean() <= (1 + MATCH_MARGIN) * u.mean()
        beats = a.mean() < u.mean() and p < 0.05
        loses = a.mean() > u.mean() and p < 0.05
        sec = []
        for k in SECONDARY:
            x, y = per_seed(runs['ARLGOS-v2'], k), per_seed(runs['Unicorn'], k)
            sec.append((k, x.mean(), y.mean(), float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0))
        adj = holm([s[3] for s in sec])
        full = len(runs['ARLGOS-v2']) >= 3 and len(runs['Unicorn']) >= 3
        verdict = {'arlgos': float(a.mean()), 'unicorn': float(u.mean()), 'relative': float(rel), 'p': p,
                   'better_seeds': int((diff < 0).sum()), 'match': bool(match), 'beats': bool(beats),
                   'loses': bool(loses), 'final': full,
                   'secondary': {k: {'arlgos': float(x), 'unicorn': float(y), 'p': pk, 'p_holm': h}
                                 for (k, x, y, pk), h in zip(sec, adj)}}
        md += ['', '## Pre-registered comparison{}'.format('' if full else
                                                           ' (PROVISIONAL: fewer than 3 training seeds per method)'),
               '', 'Primary, queue length: ARLGOS-v2 {:.3f} vs Unicorn {:.3f} ({:+.1f} %), ARLGOS-v2 lower on {} of {} '
               'test seeds, Wilcoxon p = {:.4f}.'.format(a.mean(), u.mean(), 100 * rel, verdict['better_seeds'],
                                                         len(a), p),
               '* Match (within {:.0f} %): **{}**'.format(100 * MATCH_MARGIN, 'yes' if match else 'no'),
               '* Beats (lower, p < 0.05): **{}**{}'.format('yes' if beats else 'no',
                                                             '; Unicorn is significantly lower' if loses else ''), '',
               '| Secondary | ARLGOS-v2 | Unicorn | Difference | p | p (Holm) |', '|---|---|---|---|---|---|']
        for (k, x, y, pk), h in zip(sec, adj):
            md.append('| {} | {:.3f} | {:.3f} | {:+.1f} % | {:.4f} | {:.4f} |'.format(LABEL[k], x, y,
                                                                                100 * (x - y) / y, pk, h))
        md += ['', 'The 10 test seeds replay {} with different driver randomness; the training seeds are the'.format(
            'one fixed traffic day' if args.demand == 'published' else 'ten different traffic days'),
               'other source of variation (3 per method). Read p-values with that in mind (docs/T15F.md section 2).']
    else:
        missing = [m for m in runs if not runs[m]]
        md += ['', '**No comparison yet:** no tested {} run in results/T15F/runs/.'.format(' or '.join(missing))]
    os.makedirs(OUT, exist_ok=True)
    stem = 'fenglin_h2h' + ('_heldout' if args.demand == 'heldout' else '') + args.tag
    with open(os.path.join(OUT, stem + '.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(OUT, stem + '.json'), 'w') as f:
        json.dump({'demand': args.demand, 'rows': rows, 'verdict': verdict, 'copied': copied,
                   'runs': {m: sorted(v) for m, v in runs.items()}}, f, indent=1)
    print('\n'.join(md))
    print('\nwrote', os.path.join(OUT, stem + '.md'))


if __name__ == '__main__':
    main()
