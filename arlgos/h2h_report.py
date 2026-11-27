"""ARLGOS (T15A): ARLGOS-v2 vs the supervisor's Unicorn rerun on one benchmark network, on one
machine (the lab PC), with the scoring fixed before any result. Generalises arlgos/fenglin_report.py
(T15F) to every network of the Unicorn paper; fenglin_report.py itself is unchanged.

    python arlgos/h2h_report.py --net cologne_network_8
    python arlgos/h2h_report.py --all                 # every network with runs + results/h2h/overview.md
    python arlgos/h2h_report.py --selftest

1. Collects every tested run into results/h2h/runs/<run>/ (test.json, test_sampled.json, best.json,
   val csv, config.json, import.json). That folder is small and travels through git.
     ARLGOS-v2:          Train_MATSC/arlgos_labpc_<net>_seed<k>   (--arlgos_prefix)
     Unicorn (rerun):    Train_MATSC/unicornsup_<net>_seed<k>     (--unicorn_prefix; arlgos/import_unicorn_runs.py
                                                                   + arlgos/eval_unicorn.py)
     Unicorn-multiple:   Train_MATSC/unicornsupmulti_<net>_seed<k> (an extra row, not tested)
2. Writes results/h2h/<net>.{md,json}; --all also writes results/h2h/overview.md.

Pre-registered scoring (docs/T15A.md section 3; the same as T15F, docs/T15F.md section 2):
  * metrics: the paper's six, from arlgos/version_check.episode_metrics, on Unicorn's 10 test seeds
    (1000, 1100, ..., 1900), 3600 s each; both methods greedy; checkpoints chosen on validation
    seeds 5042-5046 and tested once;
  * per test seed, a method's value = the mean over its training seeds;
  * primary: queue length. Two-sided Wilcoxon signed-rank over the 10 paired test seeds;
    "match" = ARLGOS-v2 <= 1.10 x Unicorn; "beats" = ARLGOS-v2 lower with p < 0.05;
    across networks, the headline "beats on N networks" uses Holm over the networks' p-values;
  * secondary: intersection delay and trip delay, Wilcoxon, Holm over the two;
  * speed, completion and trip time: reported, not tested.
The non-RL rows (fixed-time, greedy, max-pressure) come from results/nonrl_probe_labpc/<net>.json,
the lab PC's own runs (arlgos/run_labpc_queue.ps1 runs them), never the laptop's.
The paper's row is printed as reference only (fairness rule 1); a run trained with other signal
timings than this fork's (import.json "mismatches") is flagged in the table.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(FORK, 'results', 'h2h')
# the lab PC's own non-RL runs (arlgos/nonrl_probe.ps1 -OutRoot results\nonrl_probe_labpc): one machine per table
NONRL = os.path.join(FORK, 'results', 'nonrl_probe_labpc')
KEYS = ('queue', 'speed', 'delay', 'completion', 'trip_time', 'trip_delay')
LABEL = {'queue': 'Queue (veh)', 'speed': 'Speed (m/s)', 'delay': 'Int. delay (s)',
         'completion': 'Completion (veh/s)', 'trip_time': 'Trip time (s)', 'trip_delay': 'Trip delay (s)'}
PRIMARY, SECONDARY = 'queue', ('delay', 'trip_delay')
MATCH_MARGIN = 0.10
METHODS = ('ARLGOS-v2', 'Unicorn')
PREFIXES = {'ARLGOS-v2': 'arlgos_labpc', 'Unicorn': 'unicornsup', 'Unicorn-multiple': 'unicornsupmulti'}
COPY = (('eval/test.json', 'test.json'), ('eval/test_sampled.json', 'test_sampled.json'),
        ('eval/best.json', 'best.json'), ('eval/val.csv', 'val_unicorn.csv'), ('val.csv', 'val.csv'),
        ('config.json', 'config.json'), ('eval/config.json', 'eval_config.json'), ('done.json', 'done.json'),
        ('import.json', 'import.json'))


def holm(ps):
    from fenglin_report import holm as _holm          # the T15F function, unchanged
    return _holm(ps)


def is_pipeline_check(run_dir):
    p = os.path.join(run_dir, 'eval', 'config.json')
    if os.path.exists(p):
        with open(p) as f:
            return bool(json.load(f).get('pipeline_check'))
    return False


def collect(net, prefixes, train_root, out, include_checks=False):
    """Copy the small result files of every tested run of `net` into <out>/runs/<run>/."""
    copied = []
    for prefix in prefixes.values():
        for run in sorted(glob.glob(os.path.join(train_root, '{}_{}_seed*'.format(prefix, net)))):
            if not os.path.exists(os.path.join(run, 'eval', 'test.json')):
                continue
            if is_pipeline_check(run) and not include_checks:
                continue                                 # shortened test: not a result
            dst = os.path.join(out, 'runs', os.path.basename(run))
            os.makedirs(dst, exist_ok=True)
            for src, name in COPY:
                s = os.path.join(run, src)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(dst, name))
            copied.append(os.path.basename(run))
    return copied


def load(net, prefixes, out):
    runs, sampled, imports = {m: {} for m in prefixes}, {}, {}
    for method, prefix in prefixes.items():
        for d in sorted(glob.glob(os.path.join(out, 'runs', '{}_{}_seed*'.format(prefix, net)))):
            name = os.path.basename(d)
            with open(os.path.join(d, 'test.json')) as f:
                runs[method][name] = json.load(f)
            sp = os.path.join(d, 'test_sampled.json')
            if os.path.exists(sp):
                with open(sp) as f:
                    sampled[name] = json.load(f)
            ip = os.path.join(d, 'import.json')
            if os.path.exists(ip):
                with open(ip) as f:
                    imports[name] = json.load(f)
    return runs, sampled, imports


def per_seed(group, key):
    """(n_test_seeds,) mean over the group's training seeds of each test seed's value."""
    import numpy as np
    return np.mean(np.array([r['per_seed'][key] for r in group.values()], dtype=float), axis=0)


def verdict(arl, uni):
    """The pre-registered comparison of two run groups (each {name: test.json}); the same
    arithmetic as fenglin_report.main."""
    import numpy as np
    from scipy.stats import wilcoxon
    a, u = per_seed(arl, PRIMARY), per_seed(uni, PRIMARY)
    p = float(wilcoxon(a, u).pvalue) if np.any(a != u) else 1.0
    diff = a - u
    rel = (a.mean() - u.mean()) / u.mean()
    match = a.mean() <= (1 + MATCH_MARGIN) * u.mean()
    beats = a.mean() < u.mean() and p < 0.05
    loses = a.mean() > u.mean() and p < 0.05
    sec = []
    for k in SECONDARY:
        x, y = per_seed(arl, k), per_seed(uni, k)
        sec.append((k, x.mean(), y.mean(), float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0))
    adj = holm([s[3] for s in sec])
    full = len(arl) >= 3 and len(uni) >= 3
    return {'arlgos': float(a.mean()), 'unicorn': float(u.mean()), 'relative': float(rel), 'p': p,
            'better_seeds': int((diff < 0).sum()), 'n_seeds': int(len(a)), 'match': bool(match),
            'beats': bool(beats), 'loses': bool(loses), 'final': full,
            'secondary': {k: {'arlgos': float(x), 'unicorn': float(y), 'p': pk, 'p_holm': h}
                          for (k, x, y, pk), h in zip(sec, adj)}}


def report(net, prefixes=None, train_root=None, out=OUT, include_checks=False, collect_runs=True):
    """Collect, score and write results/h2h/<net>.{md,json}; returns the json dict."""
    import numpy as np
    from paper_tables import NETWORKS, ROWS, comparator
    from paper_compare import load_nonrl
    prefixes = dict(PREFIXES, **(prefixes or {}))
    train_root = train_root or os.path.join(FORK, 'Train_MATSC')
    copied = collect(net, prefixes, train_root, out, include_checks) if collect_runs else []
    runs, sampled, imports = load(net, prefixes, out)
    seed_lists = {tuple(r['seeds']) for group in list(runs.values()) + [sampled] for r in group.values()}
    if len(seed_lists) > 1:                              # pairing by position needs one seed list
        raise SystemExit('{}: runs were tested on different seed lists: {}'.format(net, sorted(seed_lists)))

    rows = []                                            # (label, n runs, {key: mean})
    for method in ('ARLGOS-v2', 'Unicorn', 'Unicorn-multiple'):
        if runs.get(method):
            label = {'ARLGOS-v2': '**ARLGOS-v2** (greedy)', 'Unicorn': '**Unicorn, supervisor rerun** (greedy)',
                     'Unicorn-multiple': 'Unicorn-multiple, supervisor rerun (greedy; reported, not tested)'}[method]
            rows.append((label, len(runs[method]), {k: float(per_seed(runs[method], k).mean()) for k in KEYS}))
    uni_sampled = {k: v for k, v in sampled.items() if k in runs['Unicorn']}
    if uni_sampled:
        rows.append(('Unicorn, supervisor rerun (sampled, as evaluator_rl.py)', len(uni_sampled),
                     {k: float(per_seed(uni_sampled, k).mean()) for k in KEYS}))
    nonrl = load_nonrl(os.path.join(NONRL, '{}.json'.format(net)), net)
    for agent, label in (('FIXED', 'Fixed-time'), ('GREEDY', 'Greedy'), ('PRESSURE', 'Max-pressure')):
        if agent in nonrl:
            rows.append((label, 1, {k: float(nonrl[agent][k]['seed_mean']) for k in KEYS}))

    title = NETWORKS.get(net, ('', net))[1]
    md = ['# T15A: ARLGOS-v2 vs Unicorn on {} (`{}`), lab PC'.format(title, net), '',
          'Unicorn\'s 10 test seeds (1000, 1100, ..., 1900), 3600 s each, SUMO 1.28.0, one machine. Each cell: mean',
          'over the test seeds of the mean over a method\'s training seeds. Every row is scored by the same function',
          '(`arlgos/version_check.episode_metrics`). Lower is better except speed and completion.', '']
    if include_checks and any(is_pipeline_check(os.path.join(train_root, n)) for g in runs.values() for n in g):
        md += ['**PIPELINE CHECK: shortened episodes and test seeds. Not a result.**', '']
    md += ['| Method | Runs | ' + ' | '.join(LABEL[k] for k in KEYS) + ' |', '|---|---|' + '---|' * len(KEYS)]
    for label, n, vals in rows:
        md.append('| {} | {} | {} |'.format(label, n, ' | '.join('{:.3f}'.format(vals[k]) for k in KEYS)))
    paper = {}
    if net in ROWS:
        name = comparator(net)
        paper = {k: ROWS[net][name][i][0] for i, k in enumerate(KEYS)}
        md.append('| *{}, paper {} (PDF, reference only)* | - | {} |'.format(
            name, NETWORKS[net][0], ' | '.join('*{:.2f}*'.format(paper[k]) for k in KEYS)))

    md += ['', '## Runs', '', '| Run | Chosen checkpoint | Validation queue | Test queue (greedy) | Setup differences |',
           '|---|---|---|---|---|']
    for method in ('ARLGOS-v2', 'Unicorn', 'Unicorn-multiple'):
        for name, r in runs.get(method, {}).items():
            ck = r.get('checkpoint_episode', r.get('checkpoint'))
            imp = imports.get(name, {})
            diff = '; '.join(imp.get('mismatches', [])) or ('none' if imp else '-')
            if imp:
                diff += ' (source {})'.format(imp.get('source_name', '?'))
            md.append('| {} | {} | {} | {:.3f} | {} |'.format(
                name, ck, '{:.4f}'.format(r['val_queue']) if r.get('val_queue') is not None else '-',
                float(np.mean(r['per_seed']['queue'])), diff))

    check_rows = [(label, vals['trip_time'] - vals['trip_delay']) for label, _, vals in rows]
    if paper:
        check_rows.append(('paper ' + comparator(net), paper['trip_time'] - paper['trip_delay']))
    md += ['', '## Harness check', '', 'Trip time minus trip delay = the average trip with no delay at all. No controller',
           'changes it, so within one simulator setup every row should give about the same value.', '',
           '| Row | Trip time - trip delay (s) |', '|---|---|']
    md += ['| {} | {:.1f} |'.format(label, v) for label, v in check_rows]

    v = None
    if runs['ARLGOS-v2'] and runs['Unicorn']:
        v = verdict(runs['ARLGOS-v2'], runs['Unicorn'])
        flagged = sorted(n for n in runs['Unicorn'] if imports.get(n, {}).get('mismatches'))
        v['setup_differences'] = flagged
        md += ['', '## Pre-registered comparison{}'.format('' if v['final'] else
                                                           ' (PROVISIONAL: fewer than 3 training seeds per method)'),
               '', 'Primary, queue length: ARLGOS-v2 {:.3f} vs Unicorn {:.3f} ({:+.1f} %), ARLGOS-v2 lower on {} of {} '
               'test seeds, Wilcoxon p = {:.4f}.'.format(v['arlgos'], v['unicorn'], 100 * v['relative'],
                                                         v['better_seeds'], v['n_seeds'], v['p']),
               '* Match (within {:.0f} %): **{}**'.format(100 * MATCH_MARGIN, 'yes' if v['match'] else 'no'),
               '* Beats (lower, p < 0.05): **{}**{}'.format('yes' if v['beats'] else 'no',
                                                             '; Unicorn is significantly lower' if v['loses'] else '')]
        if flagged:
            md.append('* **Setup differs** for {}: see "Setup differences" above; read this verdict with that.'.format(
                ', '.join(flagged)))
        md += ['', '| Secondary | ARLGOS-v2 | Unicorn | Difference | p | p (Holm) |', '|---|---|---|---|---|---|']
        for k, s in v['secondary'].items():
            md.append('| {} | {:.3f} | {:.3f} | {:+.1f} % | {:.4f} | {:.4f} |'.format(
                LABEL[k], s['arlgos'], s['unicorn'], 100 * (s['arlgos'] - s['unicorn']) / s['unicorn'],
                s['p'], s['p_holm']))
    else:
        missing = [m for m in METHODS if not runs[m]]
        md += ['', '**No comparison yet:** no tested {} run.'.format(' or '.join(missing))]
    os.makedirs(out, exist_ok=True)
    result = {'net': net, 'rows': rows, 'verdict': v, 'copied': copied, 'paper': paper,
              'runs': {m: sorted(x) for m, x in runs.items()}}
    with open(os.path.join(out, net + '.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(out, net + '.json'), 'w') as f:
        json.dump(result, f, indent=1)
    return result


def overview(results, out=OUT):
    """One row per network; Holm over the networks' primary p-values for the headline claim."""
    from paper_tables import NETWORKS
    scored = [r for r in results if r['verdict']]
    adj = dict(zip([r['net'] for r in scored], holm([r['verdict']['p'] for r in scored]))) if scored else {}
    md = ['# T15A overview: ARLGOS-v2 vs Unicorn (supervisor rerun), lab PC', '',
          'Queue length (veh), lower is better; per network the pre-registered test (docs/T15A.md section 3).',
          '"p (Holm, networks)" corrects the primary p-value over the networks compared.', '',
          '| Network | ARLGOS-v2 (runs) | Unicorn rerun (runs) | Difference | p | p (Holm, networks) | Match | Beats | '
          'Paper Unicorn | Note |', '|---|---|---|---|---|---|---|---|---|---|']
    beats = 0
    for r in results:
        net, v = r['net'], r['verdict']
        name = NETWORKS.get(net, ('', net))[1]
        rows = {label: (n, vals) for label, n, vals in r['rows']}
        arl = next((x for k, x in rows.items() if k.startswith('**ARLGOS')), None)
        uni = next((x for k, x in rows.items() if k.startswith('**Unicorn')), None)
        cell = lambda x: '{:.3f} ({})'.format(x[1]['queue'], x[0]) if x else '-'
        paper = '{:.2f}'.format(r['paper']['queue']) if r.get('paper') else '-'
        if v:
            b = v['beats'] and adj[net] < 0.05
            beats += b
            note = ('provisional (<3 seeds)' if not v['final'] else '') + \
                   ('; setup differs' if v.get('setup_differences') else '')
            md.append('| {} | {} | {} | {:+.1f} % | {:.4f} | {:.4f} | {} | {} | {} | {} |'.format(
                name, cell(arl), cell(uni), 100 * v['relative'], v['p'], adj[net], 'yes' if v['match'] else 'no',
                'yes' if b else 'no', paper, note.strip('; ') or ''))
        else:
            md.append('| {} | {} | {} | - | - | - | - | - | {} | not compared yet |'.format(name, cell(arl), cell(uni),
                                                                                          paper))
    md += ['', 'ARLGOS-v2 beats the supervisor\'s Unicorn rerun (Holm over networks, p < 0.05) on {} of {} networks '
               'compared.'.format(beats, len(scored))]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'overview.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    return md


def selftest():
    """Made-up runs in a temporary folder."""
    import numpy as np
    import fenglin_report
    results = []

    def ok(name, cond, detail=''):
        results.append(bool(cond))
        print('[{}] {}{}'.format('PASS' if cond else 'FAIL', name, ': ' + detail if detail else ''))

    seeds = [1000 + 100 * i for i in range(10)]
    rng = np.random.RandomState(7)

    def fake(root, name, level, method, pipeline=False, mismatches=None):
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, 'eval'), exist_ok=True)
        per = {k: list(level[k] * (1 + 0.05 * rng.randn(10))) for k in KEYS}
        t = {'run': name, 'method': method, 'seeds': seeds, 'per_seed': per, 'val_queue': level['queue'],
             'checkpoint': 'checkpoint500.pkl', 'checkpoint_episode': 900}
        with open(os.path.join(d, 'eval', 'test.json'), 'w') as f:
            json.dump(t, f)
        with open(os.path.join(d, 'eval', 'config.json'), 'w') as f:
            json.dump({'pipeline_check': pipeline}, f)
        if mismatches is not None:
            with open(os.path.join(d, 'import.json'), 'w') as f:
                json.dump({'mismatches': mismatches, 'source_name': '2026_3_5_14_UNICORN'}, f)
        return t

    base = {'queue': 0.40, 'speed': 4.0, 'delay': 15.0, 'completion': 1.2, 'trip_time': 230.0, 'trip_delay': 130.0}
    better = dict(base, queue=0.30, delay=12.0)
    tmp = tempfile.mkdtemp(prefix='arlgos_h2h_')
    try:
        net = 'shaoxing_network_7'
        train, out = os.path.join(tmp, 'Train_MATSC'), os.path.join(tmp, 'h2h')
        arl = {('arlgos_labpc_{}_seed{}'.format(net, s)): fake(train, 'arlgos_labpc_{}_seed{}'.format(net, s), better,
                                                               'ARLGOS-v2') for s in (1, 2, 3)}
        uni = {('unicornsup_{}_seed{}'.format(net, s)): fake(train, 'unicornsup_{}_seed{}'.format(net, s), base,
                                                             'Unicorn', mismatches=[]) for s in (1, 2, 3)}
        fake(train, 'arlgos_labpc_{}_seed9'.format(net), base, 'ARLGOS-v2', pipeline=True)
        r = report(net, train_root=train, out=out)
        v = r['verdict']
        ok('collects the 6 tested runs and leaves out a shortened pipeline-check test',
           sorted(r['copied']) == sorted(list(arl) + list(uni)), ', '.join(sorted(r['copied'])))
        ok('a clearly lower queue on all 10 seeds: match and beats, p < 0.05',
           v and v['match'] and v['beats'] and not v['loses'] and v['better_seeds'] == 10 and v['p'] < 0.05,
           'p = {:.4f}'.format(v['p']) if v else 'no verdict')

        # the same made-up runs through fenglin_report.main (T15F): identical verdict
        fdir = os.path.join(tmp, 'T15F')
        for name in list(arl) + list(uni):
            src = os.path.join(out, 'runs', name)
            dst = os.path.join(fdir, 'runs', name.replace('arlgos_labpc_', 'arlgos_').replace('unicornsup_', 'unicorn_'))
            shutil.copytree(src, dst)
        old_out, old_argv = fenglin_report.OUT, sys.argv
        try:
            fenglin_report.OUT, sys.argv = fdir, ['fenglin_report.py', '--no_collect']
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                fenglin_report.main()
        finally:
            fenglin_report.OUT, sys.argv = old_out, old_argv
        with open(os.path.join(fdir, 'fenglin_h2h.json')) as f:
            fv = json.load(f)['verdict']
        same = all(abs(fv[k] - v[k]) < 1e-12 for k in ('arlgos', 'unicorn', 'relative', 'p')) and \
            all(fv[k] == v[k] for k in ('better_seeds', 'match', 'beats', 'loses', 'final')) and \
            all(abs(fv['secondary'][k][x] - v['secondary'][k][x]) < 1e-12
                for k in SECONDARY for x in ('arlgos', 'unicorn', 'p', 'p_holm'))
        ok('verdict, p-values and secondary Holm equal fenglin_report.py\'s on the same runs', same)

        net2 = 'cologne_network_8'
        fake(train, 'arlgos_labpc_{}_seed1'.format(net2), dict(base, queue=0.50), 'ARLGOS-v2')   # clearly worse
        fake(train, 'unicornsup_{}_seed1'.format(net2), base, 'Unicorn', mismatches=['TELEPORT_TIME 300 (this fork: -1)'])
        r2 = report(net2, train_root=train, out=out)
        text = open(os.path.join(out, net2 + '.md'), encoding='utf-8').read()
        ok('one seed per method: PROVISIONAL; a setup difference from import.json is flagged',
           r2['verdict'] and not r2['verdict']['final'] and 'PROVISIONAL' in text
           and 'TELEPORT_TIME 300' in text and r2['verdict']['setup_differences'])
        r3 = report('grid_network_4_4', train_root=train, out=out)
        ok('a network with no runs: no verdict, "No comparison yet"', r3['verdict'] is None
           and 'No comparison yet' in open(os.path.join(out, 'grid_network_4_4.md'), encoding='utf-8').read())
        md = overview([r, r2, r3], out=out)
        ok('overview: one row per network, Holm over the compared networks',
           sum(1 for l in md if l.startswith('| ') and not l.startswith('| Network')) == 3
           and any('beats the supervisor' in l and '1 of 2' in l for l in md), md[-1])
        paper_row = [l for l in text.splitlines() if 'reference only' in l]
        ok('the paper\'s Unicorn-single row is printed as reference only', len(paper_row) == 1, paper_row[0][:60]
           if paper_row else 'missing')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return all(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--net')
    ap.add_argument('--all', action='store_true', help='every network of the paper with runs, + overview.md')
    ap.add_argument('--arlgos_prefix', default=PREFIXES['ARLGOS-v2'])
    ap.add_argument('--unicorn_prefix', default=PREFIXES['Unicorn'])
    ap.add_argument('--include_pipeline_checks', action='store_true', help='pipeline checks only (smoke runs)')
    ap.add_argument('--no_collect', action='store_true', help='use results/h2h/runs as it is')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    prefixes = {'ARLGOS-v2': args.arlgos_prefix, 'Unicorn': args.unicorn_prefix}
    if args.all:
        from paper_tables import NETWORKS
        results = [report(net, prefixes, include_checks=args.include_pipeline_checks,
                          collect_runs=not args.no_collect) for net in NETWORKS]
        print('\n'.join(overview(results)))
        print('\nwrote', os.path.join(OUT, 'overview.md'))
    elif args.net:
        r = report(args.net, prefixes, include_checks=args.include_pipeline_checks, collect_runs=not args.no_collect)
        with open(os.path.join(OUT, args.net + '.md'), encoding='utf-8') as f:
            print(f.read())
        print('wrote', os.path.join(OUT, args.net + '.md'), '' if r['verdict'] else '(no comparison yet)')
    else:
        ap.error('--net or --all')


if __name__ == '__main__':
    main()
