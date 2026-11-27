"""ARLGOS (T9/T10): checkpoint choice on validation, then the one test, for a Unicorn corridor run.

    python arlgos/eval_corridor.py --run Train_MATSC/corridor3_full_seed1 [--jobs 6]

The same rule as ARLGOS's A1 (ROADMAP fairness rule 7; docs/T9_design.md Decision 6):
  1. val : every checkpoint{N}.pkl the run saved (every 50 updates) plays the 20
           validation seeds 5042-5061 greedily (argmax over the unmasked phases);
           -> eval/val.csv. The best checkpoint has the lowest mean AQL (ties: the earlier).
  2. test: that checkpoint, once, on the 20 test seeds 1042-1061:
           greedy, with SUMO's T7 files in eval/test_outputs/<seed>/  -> eval/test.csv
           sampled from the policy, as Unicorn's evaluator_rl.py scores  -> eval/test_sampled.csv
Every episode, greedy or sampled, seeds torch with 21 + seed: Unicorn's VAE draws noise in
every forward pass, so this makes each result independent of the job scheduling.
AQL = ARLGOS's: mean camera count at the 90 decision points (CorridorEnv.aql).
Re-running skips the parts that are complete; validation saves val.csv after every episode,
so a stopped evaluation keeps its finished episodes. Sensing ('full' / 'camera') is read from the
run's config.json. The SUMO client follows UNICORN_LIBSUMO, as in training.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
VAL_SEEDS = list(range(5042, 5062))
TEST_SEEDS = list(range(1042, 1062))
SAMPLE_SEED = 21                                 # evaluator_rl.py's random_seed

_ENV = _MODEL = None


def _init(sensing):
    global _ENV, _MODEL
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    os.environ['UNICORN_NET'] = 'corridor_3'
    os.environ['UNICORN_SENSING'] = sensing
    import torch
    torch.set_num_threads(1)
    from env.corridor import CorridorEnv
    from models.Unicorn import Unicorn
    _ENV = CorridorEnv(None, test=True, sensing=sensing)
    _MODEL = Unicorn(input_dim=_ENV.tls_obs_space, agent_dim=_ENV.tls_agent_space,
                     int_vec_dim=_ENV.tls_int_attr_space, actor_lr=1e-4, critic_lr=1e-4)
    _MODEL.eval()


def play(task):
    """One episode: (checkpoint path, seed, 'greedy' | 'sampled', outputs dir or None)."""
    import numpy as np
    import torch
    from torch.distributions.categorical import Categorical
    ckpt, seed, mode, outputs = task
    env, model = _ENV, _MODEL
    if getattr(model, '_arlgos_ckpt', None) != ckpt:
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
        model._arlgos_ckpt = ckpt
    # Unicorn's forward pass draws the VAE noise eps (models/Unicorn.py: z = mu + eps * std,
    # Eqs. 5-7) even when the action is the argmax, so EVERY episode seeds torch. A result then
    # depends on (checkpoint, seed, mode) alone, not on which worker played what before it
    # (found 2026-11-06: unseeded greedy episodes changed with the job scheduling).
    torch.manual_seed(SAMPLE_SEED + seed)
    env.outputs = os.path.abspath(outputs) if outputs else None
    t0 = time.time()
    obs = env.reset_test(0, seed)
    h, done, reward, switches = None, False, 0.0, 0.0

    def tensor(d):
        return torch.tensor(np.array(list(d.values())), dtype=torch.float32)
    while not done:
        phase_vec, mask = env.get_phase_vec_mask_dict()
        int_attr = env.get_int_attr_vec_dict()
        with torch.no_grad():
            policy, h, _, _, _ = model(tensor(obs), tensor(phase_vec), tensor(int_attr), tensor(mask), h)
        if mode == 'greedy':
            action = policy.argmax(-1).reshape(-1)
        else:
            action = Categorical(policy).sample().reshape(-1)
        obs, _, done, info = env.step({tls: int(action[i]) for i, tls in enumerate(env.rl_tls_list)})
        reward += info[0]
        switches += info[1]
    if outputs:
        env.close()                 # SUMO completes tripinfo / fcd only on close or reload
    return {'checkpoint': os.path.basename(ckpt), 'seed': seed, 'mode': mode, 'aql': env.aql(),
            'reward': reward, 'switch_rate': switches / env.rl_step, 'seconds': round(time.time() - t0, 1)}


def run_tasks(tasks, sensing, jobs, on_result=None):
    """Play the tasks; the rows come back in task order. on_result(rows so far) runs after each
    one (validation saves there, so a stopped evaluation keeps its finished episodes)."""
    rows = []
    if jobs <= 1:
        _init(sensing)
        results = (play(t) for t in tasks)
        pool = None
    else:
        import multiprocessing as mp
        pool = mp.get_context('spawn').Pool(jobs, initializer=_init, initargs=(sensing,))
        results = pool.imap(play, tasks, chunksize=1)     # ordered, as pool.map was
    try:
        for r in results:
            rows.append(r)
            if on_result:
                on_result(rows)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return rows


def read_run_config(path):
    """Unicorn's config.json is 'KEY: value' lines (utils.create_config_json), not JSON."""
    cfg = {}
    with open(path) as f:
        for line in f:
            if ': ' in line:
                k, v = line.rstrip('\r\n').split(': ', 1)
                cfg[k] = v
    return cfg


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    """Whole file to path.tmp, then renamed over path: a stop mid-write leaves the old file."""
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(path + '.tmp', path)


def save_progress(path, done_rows):
    """on_result for validation: rewrite the CSV with every finished episode. A failed save
    (the file open in Excel, say) is skipped; the next episode tries again."""
    def save(new_rows):
        try:
            write_csv(path, done_rows + new_rows)
        except OSError as e:
            print('[val] progress not saved this time: {}'.format(e))
    return save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='Train_MATSC/<name>')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--smoke', action='store_true',
                    help='pipeline check only: 2 unreserved seeds each instead of val / test, '
                         'model/checkpoint.pkl if no checkpoint{N}.pkl exists, output in eval_smoke/')
    args = ap.parse_args()
    global VAL_SEEDS, TEST_SEEDS
    if args.smoke:
        VAL_SEEDS, TEST_SEEDS = [900001, 900002], [900011, 900012]
    run = os.path.abspath(os.path.join(FORK, args.run) if not os.path.isabs(args.run) else args.run)
    cfg = read_run_config(os.path.join(run, 'config.json'))
    sensing = cfg.get('SENSING', 'full')
    if cfg.get('NET_NAME') != 'corridor_3':
        raise SystemExit('{} is not a corridor_3 run'.format(run))
    out = os.path.join(run, 'eval_smoke' if args.smoke else 'eval')
    os.makedirs(out, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(run, 'model', 'checkpoint[0-9]*.pkl')),
                   key=lambda p: int(re.search(r'checkpoint(\d+)\.pkl$', p).group(1)))
    if not ckpts and args.smoke and os.path.exists(os.path.join(run, 'model', 'checkpoint.pkl')):
        ckpts = [os.path.join(run, 'model', 'checkpoint.pkl')]
    if not ckpts:
        raise SystemExit('no checkpoint{N}.pkl in ' + os.path.join(run, 'model'))

    # 1. validation
    val_path = os.path.join(out, 'val.csv')
    done = {(r['checkpoint'], int(r['seed'])) for r in read_csv(val_path)}
    tasks = [(c, s, 'greedy', None) for c in ckpts for s in VAL_SEEDS if (os.path.basename(c), s) not in done]
    if tasks:
        t0 = time.time()
        print('[val] {} episodes ({} checkpoints x {} seeds), {} jobs'.format(len(tasks), len(ckpts),
                                                                              len(VAL_SEEDS), args.jobs))
        prev = read_csv(val_path)
        rows = prev + run_tasks(tasks, sensing, args.jobs, on_result=save_progress(val_path, prev))
        write_csv(val_path, rows)
        print('[val] {:.0f} s'.format(time.time() - t0))
    rows = read_csv(val_path)
    means = {}
    for c in ckpts:
        name = os.path.basename(c)
        x = [float(r['aql']) for r in rows if r['checkpoint'] == name]
        if len(x) != len(VAL_SEEDS):
            raise SystemExit('{}: {} of {} validation episodes'.format(name, len(x), len(VAL_SEEDS)))
        means[name] = sum(x) / len(x)
    order = [os.path.basename(c) for c in ckpts]     # in update order
    best = min(order, key=lambda n: (means[n], order.index(n)))
    best_path = os.path.join(run, 'model', best)
    with open(os.path.join(out, 'best.json'), 'w') as f:
        json.dump({'checkpoint': best, 'val_aql': means[best], 'val_seeds': VAL_SEEDS,
                   'val_aql_by_checkpoint': means}, f, indent=1)
    print('[val] best {} (mean AQL {:.4f})'.format(best, means[best]))

    # 2. the one test
    for mode, name, outputs in (('greedy', 'test.csv', os.path.join(out, 'test_outputs')),
                                ('sampled', 'test_sampled.csv', None)):
        path = os.path.join(out, name)
        prev = read_csv(path)
        if len(prev) == len(TEST_SEEDS) and all(r['checkpoint'] == best for r in prev):
            continue
        tasks = [(best_path, s, mode, os.path.join(outputs, '') if outputs else None) for s in TEST_SEEDS]
        t0 = time.time()
        res = sorted(run_tasks(tasks, sensing, args.jobs), key=lambda r: r['seed'])
        write_csv(path, res)
        print('[test {}] AQL {:.4f} over {} seeds, {:.0f} s'.format(
            mode, sum(r['aql'] for r in res) / len(res), len(res), time.time() - t0))

    sys.path.insert(0, HERE)
    sys.path.insert(0, FORK)
    os.chdir(FORK)
    from common import machine_info
    from env.corridor import scenario_fingerprint
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({'run': os.path.basename(run), 'sensing': sensing, 'best_checkpoint': best,
                   'val_seeds': VAL_SEEDS, 'test_seeds': TEST_SEEDS, 'sample_seed': SAMPLE_SEED,
                   'sumo_client': 'libsumo' if os.environ.get('UNICORN_LIBSUMO') == '1' else 'traci',
                   'scenario_fingerprint': scenario_fingerprint(), 'machine': machine_info(),
                   'train_config': cfg}, f, indent=1, default=str)
    print('wrote', out)


if __name__ == '__main__':
    main()
