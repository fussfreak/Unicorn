"""ARLGOS-v2 against the Unicorn paper's published numbers, one network at a time.

Added 2026-11-22. The user's decision: the supervisor reproduces Unicorn on the lab PC; until that
rerun exists, ARLGOS-v2 is compared with the numbers printed in the paper (arlgos/paper_tables.py).

    python arlgos/paper_compare.py --net shaoxing_network_7            # prefix arlgos
    python arlgos/paper_compare.py --net cologne_network_8 --prefix arlgos

Reads every tested run Train_MATSC/<prefix>_<net>_seed*/eval/ (test.json + eval_data CSVs written
by eval_arlgos.py: each run's best-on-validation checkpoint, greedy, on Unicorn's 10 test seeds
1000, 1100, ..., 1900, 3600 s each) and writes results/paper_compare/<net>.{md,json}.

What is compared (the paper's own format):
  * pooled mean (std): every simulated second (queue, speed, intersection delay, completion) or
    every finished trip (trip time, trip delay) of the 10 test hours, all training seeds together;
  * the paper row is "Unicorn-single" (trained on this network alone, like ARLGOS-v2), or the one
    "Unicorn" row of Table II.

The verdict rule, fixed on 2026-11-22 before any of these test results was seen:
  * improvement % = (paper - ours) / paper for lower-is-better metrics, (ours - paper) / paper for
    higher-is-better ones; positive = ARLGOS-v2 is better;
  * uncertainty: the 95 % interval (t, 9 df) of ARLGOS-v2's mean over the 10 test seeds, each seed
    first averaged over the training seeds. The PDF prints no per-seed numbers, so there is no
    significance test against it;
  * "better" / "worse" = the paper's mean lies outside that interval on the better / worse side;
    "within noise" otherwise; "match" (queue only, the G4 rule) = within 10 % of the paper.
  * Primary metric: queue length.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def ci95(values):
    from scipy import stats
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float('nan')
    return m, float(stats.t.ppf(0.975, n - 1) * stats.tstd(values) / n ** 0.5)


def load_nonrl(path, net):
    """{agent: metrics} of our non-RL runs on this network (keys '<sumo>/<net>/<agent>'); {} if none."""
    nonrl = {}
    if os.path.exists(path):
        with open(path) as f:
            for name, run in json.load(f)['runs'].items():
                parts = name.split('/')
                if len(parts) < 3 or parts[-2] == net:          # another network's runs never count
                    nonrl[parts[-1]] = run['metrics']
    return nonrl


def load_runs(prefix, net, root):
    import pandas as pd
    runs = []
    for path in sorted(glob.glob(os.path.join(root, '{}_{}_seed*'.format(prefix, net), 'eval', 'test.json'))):
        eval_dir = os.path.dirname(path)
        with open(path) as f:
            result = json.load(f)
        cfg_path = os.path.join(eval_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                if json.load(f).get('pipeline_check'):
                    continue                                    # shortened test: not a result
        data = os.path.join(eval_dir, 'eval_data', '_'.join(net.split('_')[:2]) + '_ARLGOS_')
        result['traffic'] = pd.read_csv(data + 'traffic.csv')
        result['trips'] = pd.read_csv(data + 'trip.csv')
        runs.append(result)
    return runs


def compare(args):
    import pandas as pd
    from version_check import METRICS
    from paper_tables import NETWORKS, ROWS, SOURCE, comparator
    net = args.net
    if net not in ROWS:
        if getattr(args, 'skip_unknown', False):
            print('{}: not in the paper\'s tables; no comparison'.format(net))
            return None
        raise SystemExit('{} has no published rows in the paper (paper_tables.NETWORKS)'.format(net))
    root = args.root or os.path.join(REPO, 'Train_MATSC')
    runs = load_runs(args.prefix, net, root)
    if not runs:
        raise SystemExit('no tested runs {}/{}_{}_seed*/eval/test.json'.format(root, args.prefix, net))
    seed_lists = {tuple(r['seeds']) for r in runs}
    if len(seed_lists) != 1:
        raise SystemExit('runs were tested on different seed lists: {}'.format(seed_lists))
    keys = [k for k, _, _ in METRICS]

    traffic = pd.concat([r['traffic'] for r in runs], ignore_index=True)
    trips = pd.concat([r['trips'] for r in runs], ignore_index=True)
    column = {'queue': traffic['avg_queue'], 'speed': traffic['avg_speed_mps'], 'delay': traffic['avg_wait_sec'],
              'completion': traffic['number_arrived_car'], 'trip_time': trips['duration_sec'].astype(float),
              'trip_delay': trips['time_loss_sec'].astype(float)}
    pooled = {k: (float(column[k].mean()), float(column[k].std())) for k in keys}
    per_test_seed = {k: [sum(r['per_seed'][k][i] for r in runs) / len(runs) for i in range(len(runs[0]['seeds']))]
                     for k in keys}
    interval = {k: ci95(per_test_seed[k]) for k in keys}

    paper = ROWS[net]
    ref_name = comparator(net)
    ref = dict(zip(keys, paper[ref_name]))
    verdicts = {}
    for key, label, better in METRICS:
        ours, ref_mean = pooled[key][0], ref[key][0]
        gain = (ref_mean - ours) / ref_mean if better == 'lower' else (ours - ref_mean) / ref_mean
        m, half = interval[key]
        lo, hi = m - half, m + half
        if lo <= ref_mean <= hi or half != half:
            status = 'within noise'
        else:
            ours_better = (ref_mean > hi) if better == 'lower' else (ref_mean < lo)
            status = 'better' if ours_better else 'worse'
        verdicts[key] = {'ours': ours, 'paper': ref_mean, 'improvement_pct': 100 * gain, 'ci': [lo, hi],
                         'status': status, 'match_10pct': abs(gain) <= 0.10}

    probe = os.path.join(REPO, 'results', 'nonrl_probe', net + '.json')
    nonrl = load_nonrl(probe, net) if getattr(args, 'nonrl', True) else {}

    # Harness check (added 2026-11-22 after the Fenglin results, not part of the verdict rule):
    # (a) trip time - trip delay = how long the average finished trip would take with no delay at all;
    #     no controller can change it, so it should be about equal in every row of one simulator setup;
    # (b) a metric on which one of OUR non-learning controllers already beats the paper's Unicorn is
    #     flagged as suspect: there a difference between this harness and the paper's is the likelier cause.
    names = {'FIXED': 'Fixed-Time', 'GREEDY': 'Greedy', 'PRESSURE': 'Max-Pressure'}
    free_flow = {'ARLGOS-v2 (ours)': pooled['trip_time'][0] - pooled['trip_delay'][0]}
    for agent, metrics in sorted(nonrl.items()):
        free_flow['{} (ours)'.format(names.get(agent, agent))] = (metrics['trip_time']['pooled'][0]
                                                                  - metrics['trip_delay']['pooled'][0])
    for name, vals in paper.items():
        row = dict(zip(keys, vals))
        free_flow['{} (paper)'.format(name)] = row['trip_time'][0] - row['trip_delay'][0]
    for key, label, better in METRICS:
        beaten = [names.get(a, a) for a, m in sorted(nonrl.items())
                  if (m[key]['pooled'][0] < ref[key][0] if better == 'lower' else m[key]['pooled'][0] > ref[key][0])]
        verdicts[key]['simple_controllers_beat_paper'] = beaten

    table, title = NETWORKS[net]
    q = verdicts['queue']
    lead = ('**Queue length (primary): ARLGOS-v2 {:.3f} vs the paper\'s {} {:.2f}: {:+.1f} %. Interval test: {}. '
            'Within 10 % (the G4 "match" rule): {}.**'.format(
                q['ours'], ref_name, q['paper'], q['improvement_pct'], q['status'],
                'yes, a match' if q['match_10pct'] else 'no'))
    md = ['# ARLGOS-v2 vs the Unicorn paper: {} (`{}`)'.format(title, net), '', lead, '',
          'Positive % = ARLGOS-v2 better. ARLGOS-v2: {} training seed(s), each run\'s best-on-validation '
          'checkpoint, greedy, on test seeds {}. Paper: {}, {}.'.format(
              len(runs), ', '.join(str(s) for s in runs[0]['seeds']), table, SOURCE), '',
          '## ARLGOS-v2 vs {}'.format(ref_name), '',
          '| Metric | ARLGOS-v2 | 95 % interval (mean over test seeds) | Paper {} | ARLGOS-v2 better by | Verdict | Harness check |'.format(ref_name),
          '|---|---|---|---|---|---|---|']
    for key, label, better in METRICS:
        v = verdicts[key]
        flag = ('suspect: {} (ours) also beats the paper'.format(', '.join(v['simple_controllers_beat_paper']))
                if v['simple_controllers_beat_paper'] else ('ok' if nonrl else 'no non-RL rows'))
        md.append('| {} ({} is better) | {:.3f} | [{:.3f}, {:.3f}] | {:.2f} | {:+.1f} % | {} | {} |'.format(
            label, better, v['ours'], v['ci'][0], v['ci'][1], v['paper'], v['improvement_pct'], v['status'], flag))
    md += ['', '## Is this harness the paper\'s harness? (check added after the first results)', '',
           'Trip time minus trip delay is how long the average finished trip would take with no delay at all. '
           'No controller can change it, so within one simulator setup it should be about the same in every row. '
           'A clear gap between our rows and the paper\'s means the two setups differ (SUMO version, vehicle or '
           'route settings), and then speed, trip time and trip delay cannot be compared with the PDF.', '',
           '| Row | Trip time - trip delay (s) |', '|---|---|']
    md += ['| {} | {:.1f} |'.format(name, value) for name, value in free_flow.items()]
    md += ['', '"Harness check: suspect" in the first table = one of our own non-learning controllers already '
           'beats the paper\'s Unicorn on that metric, which points to a harness difference rather than a real gain.']
    md += ['', '## Every row in the paper\'s format: pooled mean (std)', '',
           '| Row | Source | ' + ' | '.join(label for _, label, _ in METRICS) + ' |',
           '|---|---|' + '---|' * len(METRICS),
           '| **ARLGOS-v2 ({} seeds pooled)** | ours, this harness | {} |'.format(
               len(runs), ' | '.join('**{:.2f}** ({:.2f})'.format(*pooled[k]) for k in keys))]
    for r in runs:
        md.append('| ARLGOS-v2 {} (ep {}) | ours, this harness | {} |'.format(
            r['run'].split('_')[-1], r.get('checkpoint_episode'),
            ' | '.join('{:.2f} ({:.2f})'.format(*r['pooled'][k]) for k in keys)))
    for agent, metrics in sorted(nonrl.items()):
        md.append('| {} | ours, same harness and seeds | {} |'.format(
            names.get(agent, agent), ' | '.join('{:.2f} ({:.2f})'.format(*metrics[k]['pooled']) for k in keys)))
    for name, vals in paper.items():
        bold = name == ref_name
        md.append('| {} | paper, {} | {} |'.format(
            '**{}**'.format(name) if bold else name, table,
            ' | '.join(('**{:.2f}** ({:.2f})' if bold else '{:.2f} ({:.2f})').format(m, s) for m, s in vals)))
    md += ['', '## Rule and limits (fixed before the test results)', '',
           '* "better" / "worse": the paper\'s mean lies outside ARLGOS-v2\'s 95 % interval over the 10 test '
           'seeds (each averaged over training seeds); otherwise "within noise". "Match" (queue) = within 10 %.',
           '* The PDF has no per-seed numbers, so there is no p-value against it, and its values are rounded '
           'to 2 decimals (a queue of 0.37 can be 0.365-0.375).',
           '* The paper does not state its SUMO version, its number of training runs, or whether test actions '
           'are sampled; ARLGOS-v2 is tested greedily on SUMO 1.28. On Cologne8 SUMO 1.16 vs 1.28 moved every '
           'metric by less than 5 % (T3).',
           '* The paper\'s rows are not rerun here; the supervisor\'s Unicorn rerun replaces them when it exists.',
           '', 'Runs: ' + ', '.join('{} (checkpoint ep {}, val {:.4f})'.format(
               r['run'], r.get('checkpoint_episode'), r.get('val_queue') or float('nan')) for r in runs)]

    out_dir = args.out or os.path.join(REPO, 'results', 'paper_compare')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, net + args.tag)
    result = {'net': net, 'prefix': args.prefix, 'paper_source': SOURCE, 'paper_table': table,
              'comparator': ref_name, 'runs': [r['run'] for r in runs], 'seeds': runs[0]['seeds'],
              'pooled': pooled, 'per_test_seed': per_test_seed, 'verdicts': verdicts, 'paper_rows': paper,
              'free_flow_trip_s': free_flow}
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(base + '.json', 'w') as f:
        json.dump(result, f, indent=1)
    if not args.quiet:
        print('\n'.join(md))
        print('\nwrote', base + '.md')
    return result


def selftest():
    """Two made-up runs with known answers (no simulation). Prints one PASS/FAIL line per check."""
    import tempfile
    import pandas as pd
    seeds = [1000, 1100, 1200]
    # per run and test seed: (queue, speed, delay, completion) constant over 4 steps; trips: (duration, loss)
    design = {'seed1': [(0.30, 3.50, 25.0, 1.0), (0.31, 3.60, 25.2, 1.0), (0.29, 3.55, 24.8, 1.0)],
              'seed2': [(0.32, 3.52, 25.1, 1.0), (0.30, 3.58, 25.0, 1.0), (0.30, 3.50, 24.9, 1.0)]}
    with tempfile.TemporaryDirectory() as root:
        for run, per_seed in design.items():
            eval_dir = os.path.join(root, 'fake_shaoxing_network_7_{}'.format(run), 'eval')
            os.makedirs(os.path.join(eval_dir, 'eval_data'))
            traffic, trips = [], []
            for ep, (q, v, d, c) in enumerate(per_seed):
                traffic += [{'episode': ep, 'avg_queue': q, 'avg_speed_mps': v, 'avg_wait_sec': d,
                             'number_arrived_car': c} for _ in range(4)]
                trips += [{'episode': ep, 'duration_sec': 230.0, 'time_loss_sec': 100.0 + ep} for _ in range(2)]
            pd.DataFrame(traffic).to_csv(os.path.join(eval_dir, 'eval_data', 'shaoxing_network_ARLGOS_traffic.csv'))
            pd.DataFrame(trips).to_csv(os.path.join(eval_dir, 'eval_data', 'shaoxing_network_ARLGOS_trip.csv'))
            per = {'queue': [x[0] for x in per_seed], 'speed': [x[1] for x in per_seed],
                   'delay': [x[2] for x in per_seed], 'completion': [x[3] for x in per_seed],
                   'trip_time': [230.0] * 3, 'trip_delay': [100.0 + i for i in range(3)]}
            with open(os.path.join(eval_dir, 'test.json'), 'w') as f:
                json.dump({'run': 'fake_shaoxing_network_7_' + run, 'seeds': seeds, 'per_seed': per,
                           'pooled': {k: [sum(x) / 3, 0.0] for k, x in per.items()},
                           'checkpoint_episode': 1, 'val_queue': 0.3}, f)
        r = compare(argparse.Namespace(net='shaoxing_network_7', prefix='fake', tag='', root=root,
                                       out=os.path.join(root, 'out'), quiet=True, nonrl=False))
        # the non-RL file: only this network's runs count (added 2026-11-23, T15)
        probe = os.path.join(root, 'probe.json')
        with open(probe, 'w') as f:
            json.dump({'runs': {'1.28.0/cologne_network_8/GREEDY': {'metrics': 'c'},
                                '1.28.0/shaoxing_network_7/GREEDY': {'metrics': 's'},
                                '1.28.0/shaoxing_network_7/FIXED': {'metrics': 'f'}}}, f)
        nonrl_ok = (load_nonrl(probe, 'cologne_network_8') == {'GREEDY': 'c'}
                    and load_nonrl(probe, 'shaoxing_network_7') == {'GREEDY': 's', 'FIXED': 'f'}
                    and load_nonrl(os.path.join(root, 'missing.json'), 'x') == {})
    from nonrl_probe import select
    fake = {('1.16.0', 'cologne_network_8', 'GREEDY'): 1, ('1.28.0', 'cologne_network_8', 'GREEDY'): 2,
            ('1.28.0', 'shaoxing_network_7', 'GREEDY'): 3}
    select_ok = (select(fake, 'cologne_network_8', '1.28.0') == {('1.28.0', 'cologne_network_8', 'GREEDY'): 2}
                 and select(fake) == fake and len(select(fake, sumo='1.28.0')) == 2)
    v = r['verdicts']
    checks = [
        ('pooled_queue', abs(r['pooled']['queue'][0] - 0.30333333) < 1e-6),
        ('queue_better', v['queue']['status'] == 'better' and abs(v['queue']['improvement_pct'] - 18.018018) < 1e-3),
        ('speed_within_noise', v['speed']['status'] == 'within noise'),
        ('delay_worse', v['delay']['status'] == 'worse' and v['delay']['improvement_pct'] < 0),
        ('completion_worse_higher_is_better', v['completion']['status'] == 'worse'),   # 1.0 < 1.20, zero spread
        ('trip_delay_better', v['trip_delay']['status'] == 'better'),                 # 101 < 104.55
        ('comparator', r['comparator'] == 'Unicorn-single'),
        ('free_flow_trip', abs(r['free_flow_trip_s']['ARLGOS-v2 (ours)'] - 129.0) < 1e-9
         and abs(r['free_flow_trip_s']['Unicorn-single (paper)'] - (233.45 - 104.55)) < 1e-9),
        ('nonrl_this_net_only', nonrl_ok),
        ('nonrl_probe_select', select_ok),
    ]
    for name, ok in checks:
        print('[{}] paper_compare_{}'.format('PASS' if ok else 'FAIL', name))
    return all(ok for _, ok in checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net')
    ap.add_argument('--prefix', default='arlgos')
    ap.add_argument('--tag', default='', help='suffix for the output name')
    ap.add_argument('--root', default=None, help='folder holding the runs (default Train_MATSC)')
    ap.add_argument('--out', default=None, help='output folder (default results/paper_compare)')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--skip_unknown', action='store_true', help='exit quietly for a network the paper lacks')
    ap.add_argument('--selftest', action='store_true', help='checks on made-up runs; exit 1 on a failure')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.net:
        ap.error('--net is required')
    compare(args)


if __name__ == '__main__':
    main()
