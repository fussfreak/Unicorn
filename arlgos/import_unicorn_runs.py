"""ARLGOS (T15A): bring the supervisor's Unicorn runs (original marmotlab/Unicorn code) into this
fork, so that arlgos/eval_unicorn.py re-scores them with ARLGOS-v2's rule on the same machine.

    python arlgos/import_unicorn_runs.py --src "D:\\supervisor\\Unicorn\\Train_MATSC" --dry_run   # list only
    python arlgos/import_unicorn_runs.py --src "D:\\supervisor\\Unicorn\\Train_MATSC"             # copy
    python arlgos/import_unicorn_runs.py --selftest

What counts as a run: a folder (the --src folder itself, or one up to 3 levels below it) holding a
config.json in Unicorn's own 'KEY: value' format (utils.create_config_json) and a model/ folder
with checkpoint{N}.pkl files. The original code names a run <year>_<month>_<day>_<hour>_UNICORN
and saves a checkpoint every SAVE_MODEL_STEP = 500 updates; the network is the config's NET_NAME.

What it does, for each run found:
  * single-network run (CO_TRAIN False) -> Train_MATSC/unicornsup_<net>_seed<k>/
  * jointly trained run (CO_TRAIN True, the paper's "Unicorn-multiple") -> one copy per network in
    its ALL_DATASETS, Train_MATSC/unicornsupmulti_<net>_seed<k>/, whose config.json has NET_NAME
    set to that network (the source's is kept as config_source.json);
  * copies model/checkpoint{N}.pkl and config.json; never writes anything inside the source;
  * k: runs of one network are numbered in the order they were trained (the date in the folder
    name, else the folder's time); a run imported before keeps its number;
  * writes import.json: source path, file sizes and SHA-256 hashes, checkpoints, last update, and
    the run's signal timing next to this fork's for that network. A run trained with other
    green / yellow / teleport times, episode length or observation type is a different setup:
    it is listed under "mismatches" and the head-to-head report prints it.
Re-running is safe: files already copied with the same hash are skipped.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, FORK)

PREFIX = {'single': 'unicornsup', 'multiple': 'unicornsupmulti'}
MAX_DEPTH = 3
# settings that make a training a different setup if they differ from this fork's
COMPARED = ('GREEN_DURATION', 'YELLOW_DURATION', 'TELEPORT_TIME', 'MAX_SUMO_STEP', 'OBS_TYPE')
INFO = ('MAX_EPISODE', 'SAVE_MODEL_STEP', 'NUM_META_AGENTS', 'UPDATE_TYPE', 'SEED', 'RANDOM_SEED')


def read_config(path):
    """Unicorn's config.json: 'KEY: value' lines. Returns None for a real JSON file (an ARLGOS-v2
    run's config.json), which is not a Unicorn run."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    if text.lstrip().startswith('{'):
        return None
    cfg = {}
    for line in text.splitlines():
        if ': ' in line:
            k, v = line.split(': ', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def ours(net):
    """This fork's values for `net` (parameters.py: FAMILY_TIMING by dataset family)."""
    from parameters import DATASET_FAMILY, FAMILY_TIMING
    green, yellow, teleport = FAMILY_TIMING[DATASET_FAMILY[net]]
    return {'GREEN_DURATION': green, 'YELLOW_DURATION': yellow, 'TELEPORT_TIME': teleport,
            'MAX_SUMO_STEP': 3600, 'OBS_TYPE': 'V1'}


def same(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip("'\"") == str(b).strip("'\"")


def checkpoint_files(run):
    files = glob.glob(os.path.join(run, 'model', 'checkpoint[0-9]*.pkl'))
    files = [f for f in files if re.search(r'checkpoint(\d+)\.pkl$', f)]
    return sorted(files, key=lambda p: int(re.search(r'checkpoint(\d+)\.pkl$', p).group(1)))


def train_order(run):
    """Sort key: the date in an original-code folder name (<y>_<m>_<d>_<h>_UNICORN, not zero-padded,
    so a plain string sort puts October before March), else the config's time."""
    m = re.search(r'(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_UNICORN', os.path.basename(run))
    if m:
        return (0, tuple(int(x) for x in m.groups()), run)
    return (1, (os.path.getmtime(os.path.join(run, 'config.json')),), run)


def find_runs(src, dest):
    """Unicorn runs under src (src itself included), at most MAX_DEPTH levels down."""
    src = os.path.abspath(src)
    try:
        inside = os.path.commonpath([src, os.path.abspath(dest)]) == os.path.abspath(dest)
    except ValueError:                                # different drives
        inside = False
    if inside:
        raise SystemExit('--src {} is inside this fork\'s run folder {}; point it at the supervisor\'s runs'
                         .format(src, dest))
    runs = []
    base = src.rstrip('\\/').count(os.sep)
    for root, dirs, files in os.walk(src):
        if root.count(os.sep) - base >= MAX_DEPTH:
            dirs[:] = []
        if 'config.json' in files and os.path.isdir(os.path.join(root, 'model')) and checkpoint_files(root):
            cfg = read_config(os.path.join(root, 'config.json'))
            if cfg is not None and cfg.get('NET_NAME'):
                runs.append((root, cfg))
            dirs[:] = []                              # a run's own sub-folders hold no further runs
    return runs


def networks_of(cfg):
    if cfg.get('CO_TRAIN', 'False') == 'True':
        try:
            nets = ast.literal_eval(cfg.get('ALL_DATASETS', '[]'))
        except (ValueError, SyntaxError):
            nets = []
        return 'multiple', list(nets)
    return 'single', [cfg['NET_NAME']]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def existing_imports(dest):
    """(source path, net) -> destination folder, for every run imported before."""
    found = {}
    for p in glob.glob(os.path.join(dest, 'unicornsup*_seed*', 'import.json')):
        with open(p) as f:
            info = json.load(f)
        found[(os.path.normcase(info['source']), info['net'])] = os.path.dirname(p)
    return found


def plan(srcs, dest):
    """Every (run, network) to import, with its destination; existing imports keep their seed."""
    runs = []
    for src in srcs:
        runs += find_runs(src, dest)
    runs.sort(key=lambda rc: train_order(rc[0]))
    before = existing_imports(dest)
    used = {}
    for d in before.values():
        m = re.match(r'(.+)_seed(\d+)$', os.path.basename(d))
        used.setdefault(m.group(1), set()).add(int(m.group(2)))
    items, skipped = [], []
    for run, cfg in runs:
        variant, nets = networks_of(cfg)
        if not nets:
            skipped.append((run, 'CO_TRAIN run without a readable ALL_DATASETS'))
            continue
        for net in nets:
            if not os.path.isdir(os.path.join(FORK, 'maps', net)):
                skipped.append((run, 'network {} is not in maps/'.format(net)))
                continue
            key = (os.path.normcase(os.path.abspath(run)), net)
            if key in before:
                out = before[key]
            else:
                stem = '{}_{}'.format(PREFIX[variant], net)
                k = 1
                while k in used.get(stem, set()):
                    k += 1
                used.setdefault(stem, set()).add(k)
                out = os.path.join(dest, '{}_seed{}'.format(stem, k))
            mine = ours(net)
            settings = {key_: {'run': cfg.get(key_), 'ours': mine[key_]} for key_ in COMPARED}
            mismatches = ['{} {} (this fork: {})'.format(k_, v['run'], v['ours'])
                          for k_, v in settings.items() if v['run'] is None or not same(v['run'], v['ours'])]
            ckpts = checkpoint_files(run)
            items.append({'source': os.path.abspath(run), 'net': net, 'variant': variant, 'dest': out,
                          'checkpoints': [os.path.basename(c) for c in ckpts],
                          'last_update': int(re.search(r'(\d+)\.pkl$', ckpts[-1]).group(1)),
                          'settings': settings, 'mismatches': mismatches,
                          'info': {k_: cfg.get(k_) for k_ in INFO}, 'cfg': cfg})
    return items, skipped


def do_import(item):
    """Copy one run's checkpoints + config into item['dest']; return the import.json written."""
    run, out = item['source'], item['dest']
    os.makedirs(os.path.join(out, 'model'), exist_ok=True)
    files = []
    for name in item['checkpoints']:
        s, d = os.path.join(run, 'model', name), os.path.join(out, 'model', name)
        h = sha256(s)
        if not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s) and sha256(d) == h):
            shutil.copy2(s, d + '.tmp')
            os.replace(d + '.tmp', d)
        files.append({'name': name, 'bytes': os.path.getsize(s), 'sha256': h})
    src_cfg = os.path.join(run, 'config.json')
    if item['variant'] == 'multiple':
        shutil.copy2(src_cfg, os.path.join(out, 'config_source.json'))
        with open(src_cfg, encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
        lines = ['NET_NAME: {}'.format(item['net']) if l.startswith('NET_NAME: ') else l for l in lines]
        with open(os.path.join(out, 'config.json'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    else:
        shutil.copy2(src_cfg, os.path.join(out, 'config.json'))
    info = {'source': run, 'source_name': os.path.basename(run), 'net': item['net'], 'variant': item['variant'],
            'run': os.path.basename(out), 'checkpoints': files, 'last_update': item['last_update'],
            'config_sha256': sha256(src_cfg), 'settings': item['settings'], 'mismatches': item['mismatches'],
            'info': item['info'], 'imported': time.strftime('%Y-%m-%d %H:%M'),
            'tool': 'arlgos/import_unicorn_runs.py'}
    with open(os.path.join(out, 'import.json'), 'w') as f:
        json.dump(info, f, indent=1)
    return info


def table(items, skipped):
    md = ['| Run (source folder) | Network | Variant | Checkpoints | Last update | Imports as | Setup differences |',
          '|---|---|---|---|---|---|---|']
    for it in items:
        md.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            it['source'], it['net'], it['variant'], len(it['checkpoints']), it['last_update'],
            os.path.basename(it['dest']), '; '.join(it['mismatches']) or 'none'))
    if not items:
        md.append('| (no Unicorn runs found) | | | | | | |')
    for run, why in skipped:
        md.append('')
        md.append('Skipped {}: {}'.format(run, why))
    return md


def selftest():
    """Made-up runs in a temporary folder: discovery, numbering, CO_TRAIN, mismatches, no writes
    into the source, idempotent re-run."""
    results = []

    def ok(name, cond, detail=''):
        results.append(cond)
        print('[{}] {}{}'.format('PASS' if cond else 'FAIL', name, ': ' + detail if detail else ''))

    def make(root, name, cfg, ckpts=(500, 1000), json_cfg=False):
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, 'model'))
        with open(os.path.join(d, 'config.json'), 'w') as f:
            f.write(json.dumps(cfg) if json_cfg else ''.join('{}: {}\n'.format(k, v) for k, v in cfg.items()))
        for n in ckpts:
            with open(os.path.join(d, 'model', 'checkpoint{}.pkl'.format(n)), 'wb') as f:
                f.write(os.urandom(64))
        return d

    def snapshot(root):
        return {os.path.relpath(os.path.join(r, f), root): sha256(os.path.join(r, f))
                for r, _, fs in os.walk(root) for f in fs}

    base = {'GREEN_DURATION': 15, 'YELLOW_DURATION': 5, 'TELEPORT_TIME': -1, 'MAX_SUMO_STEP': 3600,
            'OBS_TYPE': 'V1', 'CO_TRAIN': False, 'SAVE_MODEL_STEP': 500, 'MAX_EPISODE': 3000}
    tmp = tempfile.mkdtemp(prefix='arlgos_import_')
    try:
        src, dest = os.path.join(tmp, 'supervisor', 'Train_MATSC'), os.path.join(tmp, 'fork', 'Train_MATSC')
        os.makedirs(dest)
        make(src, '2026_10_2_9_UNICORN', dict(base, NET_NAME='cologne_network_8'))          # trained second
        make(src, '2026_3_5_14_UNICORN', dict(base, NET_NAME='cologne_network_8', TELEPORT_TIME=300))  # first
        make(src, '2026_4_1_8_UNICORN', dict(base, NET_NAME='shaoxing_network_7', CO_TRAIN=True,
                                             ALL_DATASETS="['cologne_network_8', 'shaoxing_network_7']",
                                             TELEPORT_TIME=600))
        make(src, 'arlgos_run', {'net': 'cologne_network_8'}, json_cfg=True)                 # ARLGOS-v2: skip
        make(src, '2026_5_1_1_UNICORN', dict(base, NET_NAME='cologne_network_8'), ckpts=())   # no checkpoint
        before = snapshot(src)

        items, skipped = plan([src], dest)
        names = sorted(os.path.basename(i['dest']) for i in items)
        ok('finds the 3 Unicorn runs (4 imports), skips an ARLGOS-v2 run and a run without checkpoints',
           names == ['unicornsup_cologne_network_8_seed1', 'unicornsup_cologne_network_8_seed2',
                     'unicornsupmulti_cologne_network_8_seed1', 'unicornsupmulti_shaoxing_network_7_seed1'],
           ', '.join(names))
        first = [i for i in items if i['dest'].endswith('unicornsup_cologne_network_8_seed1')][0]
        ok('seeds follow the training date (March before October, not string order)',
           os.path.basename(first['source']) == '2026_3_5_14_UNICORN', os.path.basename(first['source']))
        ok('a different teleport time is a listed setup difference; equal values are not',
           first['mismatches'] == ['TELEPORT_TIME 300 (this fork: -1)']
           and [i for i in items if i['dest'].endswith('unicornsup_cologne_network_8_seed2')][0]['mismatches'] == [],
           str(first['mismatches']))

        for it in items:
            do_import(it)
        multi = os.path.join(dest, 'unicornsupmulti_cologne_network_8_seed1')
        cfg = read_config(os.path.join(multi, 'config.json'))
        ok('a CO_TRAIN run is imported per network with NET_NAME set to it, the original kept',
           cfg['NET_NAME'] == 'cologne_network_8'
           and read_config(os.path.join(multi, 'config_source.json'))['NET_NAME'] == 'shaoxing_network_7')
        copied = all(sha256(os.path.join(it['dest'], 'model', c)) == sha256(os.path.join(it['source'], 'model', c))
                     for it in items for c in it['checkpoints'])
        ok('checkpoints copied byte for byte; the source folder is unchanged',
           copied and snapshot(src) == before)

        make(src, '2026_1_1_1_UNICORN', dict(base, NET_NAME='cologne_network_8'))            # earliest, added later
        items2, _ = plan([src], dest)
        by_source = {(os.path.basename(i['source']), i['net']): os.path.basename(i['dest']) for i in items2}
        ok('re-run: earlier imports keep their seed; a run added later gets the next free one',
           by_source[('2026_3_5_14_UNICORN', 'cologne_network_8')] == 'unicornsup_cologne_network_8_seed1'
           and by_source[('2026_1_1_1_UNICORN', 'cologne_network_8')] == 'unicornsup_cologne_network_8_seed3',
           str(sorted(by_source.values())))
        try:
            find_runs(dest, dest)
            refused = False
        except SystemExit:
            refused = True
        ok('refuses a --src inside this fork\'s own run folder', refused)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return all(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', action='append', default=[], help="the supervisor's Train_MATSC (repeatable)")
    ap.add_argument('--dest', default=os.path.join(FORK, 'Train_MATSC'))
    ap.add_argument('--dry_run', action='store_true', help='list what would be imported; copy nothing')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.src:
        ap.error('--src is required')
    for s in args.src:
        if not os.path.isdir(s):
            raise SystemExit('--src {}: no such folder'.format(s))
    items, skipped = plan(args.src, args.dest)
    print('\n'.join(table(items, skipped)))
    if args.dry_run:
        return
    for it in items:
        info = do_import(it)
        print('imported {} -> {} ({} checkpoints{})'.format(
            info['source_name'], info['run'], len(info['checkpoints']),
            '; setup differs: ' + '; '.join(info['mismatches']) if info['mismatches'] else ''))
    print('{} import(s) in {}'.format(len(items), args.dest))


if __name__ == '__main__':
    main()
