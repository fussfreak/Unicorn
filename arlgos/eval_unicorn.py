"""ARLGOS (T15F): checkpoint choice on validation, then the one test, for a Unicorn run on a
benchmark network (Fenglin first). The same rule ARLGOS-v2 follows (ROADMAP fairness rule 7).

    python arlgos/eval_unicorn.py --run Train_MATSC/unicorn_shaoxing_network_7_seed1 [--jobs 6]

  1. val : every checkpoint{N}.pkl the run saved (every 50 updates) plays the validation seeds
           5042-5046 greedily (argmax over the junction's real phases)          -> eval/val.csv
           score = the decision-point queue ARLGOS-v2's own validation uses
           (env/arlgos_env.ArlgosEnv.mean_queue: halting vehicles per incoming lane, whole lane,
           averaged over the decision points). Best = lowest mean; ties: the earlier checkpoint.
  2. test: that checkpoint, once, on Unicorn's 10 test seeds (1000, 1100, ..., 1900):
           greedy  -> eval/test.json, test.md, eval_data/*.csv
           sampled -> eval/test_sampled.json (evaluator_rl.py's way: Categorical sample,
                      re-drawn while it names a padded phase)
           The six metrics come from arlgos/version_check.episode_metrics on the environment's
           own per-second and per-trip records: the function that scores ARLGOS-v2
           (arlgos/eval_arlgos.py) and the non-RL controllers (T3). One function, one file kind.

Every episode seeds torch with 21 + seed (Unicorn's VAE draws noise in each forward pass), so a
result depends on (checkpoint, seed, mode) only, not on which worker played what before.
Demand (published / held-out) follows UNICORN_DEMAND, which must equal the run's own setting
(read from its config.json when the run recorded it). The SUMO client follows UNICORN_LIBSUMO.
Re-running skips finished parts; validation saves after every episode.

--smoke: pipeline check only (2 unreserved seeds, short episodes, model/checkpoint.pkl allowed,
output in eval_smoke/).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
# Python's XML parser BEFORE libsumo (env/arlgos_env.py has the same line): on Windows libsumo
# loads its own expat, after which ElementTree fails in collect_trip_data ("No module named expat").
import xml.parsers.expat  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_corridor import read_csv, read_run_config, save_progress, write_csv  # noqa: E402

VAL_SEEDS = [5042, 5043, 5044, 5045, 5046]          # = driver_arlgos.py --val_seeds default
SAMPLE_SEED = 21                                    # evaluator_rl.py's random_seed
METRIC_KEYS = ('queue', 'speed', 'delay', 'completion', 'trip_time', 'trip_delay')

_ENV = _MODEL = None
_SIM_SECONDS = None


def decision_queue(env):
    """The quantity ArlgosEnv._scan appends to queue_series at every observation: the mean over
    the RL junctions' configured incoming lanes of SUMO's halting number for the whole lane."""
    from env.sumo_api import traci
    whole = [traci.lane.getLastStepHaltingNumber(lane)
             for t in env.rl_tls_list for lane in env.tls_dict[t].incoming_lane_list]
    return float(sum(whole)) / len(whole)


def measure_at_observe(env):
    """Record decision_queue(env) in env.decision_queues each time the env observes, i.e. at the
    moment ArlgosEnv records its own (inside observe(), before step() checks the clock and closes
    SUMO at the end of the episode). The instance attribute shadows the class method."""
    observe = env.observe
    env.decision_queues = []

    def observe_and_measure(*a, **kw):
        obs = observe(*a, **kw)
        env.decision_queues.append(decision_queue(env))
        return obs
    env.observe = observe_and_measure
    return env


def _init(net, sim_seconds, server_base):
    global _ENV, _MODEL, _SIM_SECONDS
    import multiprocessing as mp
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    os.environ['UNICORN_NET'] = net
    import torch
    torch.set_num_threads(1)
    from common import use_env_sumo
    use_env_sumo()
    from utils import set_env
    from models.Unicorn import Unicorn
    ident = mp.current_process()._identity
    server = server_base + (ident[0] if ident else 0)   # own TraCI port and own SUMO files
    _ENV = measure_at_observe(set_env(server_number=server, test=True))
    _SIM_SECONDS = sim_seconds
    if sim_seconds:
        _ENV.max_test_step = sim_seconds
    _MODEL = Unicorn(input_dim=_ENV.tls_obs_space, agent_dim=_ENV.tls_agent_space,
                     int_vec_dim=_ENV.tls_int_attr_space, actor_lr=1e-4, critic_lr=1e-4)
    _MODEL.eval()


def play(task):
    """One episode: (checkpoint path, seed, 'greedy' | 'sampled', index, record). With record,
    the per-second and per-trip rows come back too (episode = index)."""
    import numpy as np
    import torch
    from torch.distributions.categorical import Categorical
    ckpt, seed, mode, index, record = task
    env, model = _ENV, _MODEL
    if getattr(model, '_arlgos_ckpt', None) != ckpt:
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
        model._arlgos_ckpt = ckpt
    torch.manual_seed(SAMPLE_SEED + seed)
    t0 = time.time()
    env.traffic_data, env.trip_data = [], []
    # evaluator_rl.Evaluator.evaluate + reset, call for call
    env.curr_episode = index
    if env.set_world(seed=seed) != 0:
        raise RuntimeError('SUMO did not start (seed {}, server {})'.format(seed, env.server_number))
    env.reset_vars()
    for tls in env.tls_list:
        env.tls_dict[tls].set_green_phase(0, env.sumo_steps_green_phase)
    env.decision_queues = []
    obs = env.observe()
    h, done, steps, reward, switches = None, False, 0, 0.0, 0.0

    def tensor(d):
        return torch.tensor(np.array(list(d.values())), dtype=torch.float32)
    while not done:
        phase_vec, mask = env.get_phase_vec_mask_dict()
        int_attr = env.get_int_attr_vec_dict()
        with torch.no_grad():
            policy, h, _, _, _ = model(tensor(obs), tensor(phase_vec), tensor(int_attr), tensor(mask), h)
        if mode == 'greedy':
            action = policy.argmax(-1).reshape(-1)       # padded phases have probability 0
        else:
            dist = Categorical(policy)
            action = dist.sample().reshape(-1)
        actions = {}
        for i, tls in enumerate(env.rl_tls_list):
            a = int(action[i])
            if mode == 'sampled':
                while a >= len(env.tls_dict[tls].action_space):    # evaluator_rl.py's re-draw
                    a = int(Categorical(policy[:, i, :].unsqueeze(0)).sample().reshape(-1)[0])
            elif a >= len(env.tls_dict[tls].action_space):
                raise RuntimeError('greedy action {} is a padded phase of {}'.format(a, tls))
            actions[tls] = a
        obs, _, done, info = env.step(actions)
        steps += 1
        reward += info[0]
        switches += info[1]
    out = {'checkpoint': os.path.basename(ckpt), 'seed': seed, 'mode': mode,
           'queue': float(np.mean(env.decision_queues)), 'decision_points': len(env.decision_queues),
           'reward': reward, 'switch_rate': switches / max(steps, 1),
           'steps': steps, 'seconds': round(time.time() - t0, 1)}
    if record:
        env.collect_trip_data()
        out['traffic'], out['trips'] = list(env.traffic_data), list(env.trip_data)
    return out


def play_safe(task):
    """play() for a worker process: libsumo's exceptions cannot be pickled back to the parent,
    so re-raise any error as a plain RuntimeError carrying the worker's traceback."""
    try:
        return play(task)
    except Exception:
        import traceback
        raise RuntimeError('episode {} failed in a worker: {}'.format(task[:4], traceback.format_exc())) from None


def run_tasks(tasks, net, jobs, sim_seconds, server_base, on_result=None):
    rows = []
    if jobs <= 1:
        _init(net, sim_seconds, server_base)
        results, pool = (play(t) for t in tasks), None
    else:
        import multiprocessing as mp
        pool = mp.get_context('spawn').Pool(jobs, initializer=_init, initargs=(net, sim_seconds, server_base))
        results = pool.imap(play_safe, tasks, chunksize=1)
    try:
        for r in results:
            rows.append(r)
            if on_result:
                on_result([{k: v for k, v in x.items() if k not in ('traffic', 'trips')} for x in rows])
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return rows


def checkpoints(run, smoke):
    ckpts = sorted(glob.glob(os.path.join(run, 'model', 'checkpoint[0-9]*.pkl')),
                   key=lambda p: int(re.search(r'checkpoint(\d+)\.pkl$', p).group(1)))
    if not ckpts and smoke and os.path.exists(os.path.join(run, 'model', 'checkpoint.pkl')):
        ckpts = [os.path.join(run, 'model', 'checkpoint.pkl')]
    if not ckpts:
        raise SystemExit('no checkpoint{N}.pkl in ' + os.path.join(run, 'model'))
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='Train_MATSC/<name>')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--server_base', type=int, default=500)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    run = os.path.abspath(os.path.join(FORK, args.run) if not os.path.isabs(args.run) else args.run)
    cfg = read_run_config(os.path.join(run, 'config.json'))
    net = cfg.get('NET_NAME')
    if not net or net == 'corridor_3':
        raise SystemExit('{}: not a benchmark-network run (NET_NAME {!r}); corridor runs use eval_corridor.py'
                         .format(run, net))
    demand = os.environ.get('UNICORN_DEMAND', 'published') or 'published'
    run_demand = cfg.get('DEMAND')
    if run_demand and run_demand != demand:
        raise SystemExit('the run trained with UNICORN_DEMAND={} but this shell has {}'.format(run_demand, demand))
    os.environ['UNICORN_NET'] = net
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    from parameters import SUMO_PARAMS
    val_seeds, test_seeds, sim_seconds = VAL_SEEDS, list(SUMO_PARAMS.TEST_SEEDS), None
    if args.smoke:
        val_seeds, test_seeds, sim_seconds = [900001, 900002], [900011, 900012], 120
    out = os.path.join(run, 'eval_smoke' if args.smoke else 'eval')
    os.makedirs(out, exist_ok=True)
    ckpts = checkpoints(run, args.smoke)

    # 1. validation
    val_path = os.path.join(out, 'val.csv')
    done = {(r['checkpoint'], int(r['seed'])) for r in read_csv(val_path)}
    tasks = [(c, s, 'greedy', 0, False) for c in ckpts for s in val_seeds if (os.path.basename(c), s) not in done]
    if tasks:
        t0 = time.time()
        print('[val] {} episodes ({} checkpoints x {} seeds), {} jobs'.format(len(tasks), len(ckpts),
                                                                              len(val_seeds), args.jobs), flush=True)
        prev = read_csv(val_path)
        rows = prev + [{k: v for k, v in r.items() if k not in ('traffic', 'trips')}
                       for r in run_tasks(tasks, net, args.jobs, sim_seconds, args.server_base,
                                          on_result=save_progress(val_path, prev))]
        write_csv(val_path, rows)
        print('[val] {:.0f} s'.format(time.time() - t0), flush=True)
    rows = read_csv(val_path)
    means, order = {}, [os.path.basename(c) for c in ckpts]
    for name in order:
        x = [float(r['queue']) for r in rows if r['checkpoint'] == name]
        if len(x) != len(val_seeds):
            raise SystemExit('{}: {} of {} validation episodes'.format(name, len(x), len(val_seeds)))
        means[name] = sum(x) / len(x)
    best = min(order, key=lambda n: (means[n], order.index(n)))
    best_path = os.path.join(run, 'model', best)
    with open(os.path.join(out, 'best.json'), 'w') as f:
        json.dump({'checkpoint': best, 'val_queue': means[best], 'val_seeds': val_seeds,
                   'val_queue_by_checkpoint': means}, f, indent=1)
    print('[val] best {} (mean decision-point queue {:.4f})'.format(best, means[best]), flush=True)

    # 2. the one test, greedy and sampled
    import pandas as pd
    from common import machine_info
    from version_check import METRICS, episode_metrics
    for mode, fname in (('greedy', 'test.json'), ('sampled', 'test_sampled.json')):
        path = os.path.join(out, fname)
        if os.path.exists(path):
            with open(path) as f:
                if json.load(f).get('checkpoint') == best:
                    continue
        t0 = time.time()
        res = sorted(run_tasks([(best_path, s, mode, i, True) for i, s in enumerate(test_seeds)], net,
                               args.jobs, sim_seconds, args.server_base), key=lambda r: r['seed'])
        traffic = pd.DataFrame([row for r in res for row in r['traffic']])
        trips = pd.DataFrame([row for r in res for row in r['trips']])
        data = os.path.join(out, 'eval_data')
        os.makedirs(data, exist_ok=True)
        prefix = '_'.join(net.split('_')[:2])
        tag = 'UNICORN' if mode == 'greedy' else 'UNICORN_sampled'
        traffic.to_csv(os.path.join(data, '{}_{}_traffic.csv'.format(prefix, tag)))
        trips.to_csv(os.path.join(data, '{}_{}_trip.csv'.format(prefix, tag)))
        per_episode, pooled = episode_metrics(traffic, trips)
        result = {'run': os.path.basename(run), 'net': net, 'method': 'Unicorn', 'mode': mode,
                  'checkpoint': best, 'val_queue': means[best], 'seeds': test_seeds, 'demand': demand,
                  'per_seed': {k: [float(per_episode[e][k]) for e in sorted(per_episode)] for k in METRIC_KEYS},
                  'pooled': {k: [float(pooled[k][0]), float(pooled[k][1])] for k in METRIC_KEYS},
                  'decision_queue': [r['queue'] for r in res], 'switch_rate': [r['switch_rate'] for r in res],
                  'episode_seconds': [r['seconds'] for r in res]}
        with open(path, 'w') as f:
            json.dump(result, f, indent=1)
        if mode == 'greedy':
            md = ['# Test: {}'.format(result['run']), '',
                  '`{}` (validation queue {:.4f}) on seeds {}, greedy, {} s episodes, demand {}.'.format(
                      best, means[best], test_seeds, _ENV.max_test_step if _ENV else 'see config', demand), '',
                  '| Metric | Mean over seeds | Pooled mean (std), as the paper reports | Better |', '|---|---|---|---|']
            for key, label, better in METRICS:
                x = result['per_seed'][key]
                md.append('| {} | {:.3f} | {:.2f} ({:.2f}) | {} |'.format(label, sum(x) / len(x), pooled[key][0],
                                                                      pooled[key][1], better))
            with open(os.path.join(out, 'test.md'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(md) + '\n')
        print('[test {}] queue {:.4f} over {} seeds, {:.0f} s'.format(
            mode, sum(result['per_seed']['queue']) / len(test_seeds), len(test_seeds), time.time() - t0), flush=True)

    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({'run': os.path.basename(run), 'net': net, 'best_checkpoint': best, 'val_seeds': val_seeds,
                   'test_seeds': test_seeds, 'sample_seed': SAMPLE_SEED, 'demand': demand,
                   'pipeline_check': bool(args.smoke), 'fcd': os.environ.get('UNICORN_FCD', '0'),
                   'sumo_client': 'libsumo' if os.environ.get('UNICORN_LIBSUMO') == '1' else 'traci',
                   'machine': machine_info(), 'train_config': cfg}, f, indent=1, default=str)
    print('wrote', out)


if __name__ == '__main__':
    main()
