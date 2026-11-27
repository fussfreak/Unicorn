"""ARLGOS (T14): train ARLGOS-v2 on one of Unicorn's benchmark networks.

    python driver_arlgos.py --net cologne_network_8 --seed 1
    python driver_arlgos.py --net cologne_network_8 --seed 1 --resume      # after a stop

One process, one simulation, no Ray: QMIX learns from a replay buffer, so it does not need
Unicorn's six parallel workers. Run several seeds side by side instead (arlgos/run_arlgos.ps1).

Output, in Train_MATSC/<name>/ (name defaults to arlgos_<net>_seed<k>):
    config.json            every setting, the machine, the graph's size
    train.csv              one row per training episode
    val.csv                one row per validation (greedy episodes on --val_seeds)
    model/best_val.pt      the policy with the lowest validation queue so far
    model/ckpt_<ep>.pt     the policy every --save_every episodes
    model/latest.pt        full training state every --latest_every episodes (for --resume)

Seeds (ROADMAP rules): training simulations draw their SUMO seed from the env's
draw_train_seed(), which never returns a test seed (1000, 1100, ..., 1900) or a validation
seed (5042-5061). Validation uses --val_seeds; the test seeds are touched only by
arlgos/eval_arlgos.py, once per configuration.

A resumed run restores the networks, optimiser, schedule, epsilon, replay buffer and all random
streams from latest.pt, and drops the log rows written after that file. It then continues exactly
as the uninterrupted run would have (arlgos/check_arlgos.py tests this).
"""
import argparse
import csv
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--net', default='cologne_network_8')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--episodes', type=int, default=1500)
    p.add_argument('--name', default='')
    p.add_argument('--server', type=int, default=None,
                   help='worker number for SUMO files and the TraCI port; default 100 + seed')
    p.add_argument('--resume', action='store_true')
    # the model (defaults = the A1 recipe; the others are ablations)
    p.add_argument('--algo', default='qmix', choices=['qmix', 'vdn', 'iql'])
    p.add_argument('--conv', default='flow', choices=['flow', 'self_nb'])
    p.add_argument('--no_inter', action='store_true', help='drop the edges between junctions')
    p.add_argument('--no_agent_id', action='store_true', help='no junction one-hot in the Q head')
    p.add_argument('--reward', default='halting', choices=['halting', 'count'])
    p.add_argument('--lane_weight', default='max', choices=['max', 'mean'],
                   help='how a phase is described per lane: best movement (T14) or average of the '
                        'movements (T15, 2026-11-23); see env/arlgos_env.py')
    # A1's hyperparameters (rl_training.py defaults + run_A1.ps1)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--memory_size', type=int, default=20000)
    p.add_argument('--target_update', type=int, default=50)
    p.add_argument('--eps_decay', type=float, default=0.82)
    p.add_argument('--eps_min', type=float, default=None,
                   help='per-junction floor; default keeps the team at A1\'s 14.3 %% (Arlgos.eps_floor)')
    p.add_argument('--pretrain_eps', type=int, default=2)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--mixer_embed', type=int, default=32)
    p.add_argument('--terminal_at_timeout', action='store_true',
                   help='store the last transition as terminal (A1 default before --bootstrap_timeout)')
    # validation and saving
    p.add_argument('--val_seeds', type=int, nargs='*', default=[5042, 5043, 5044, 5045, 5046])
    p.add_argument('--val_every', type=int, default=25)
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--latest_every', type=int, default=5)
    p.add_argument('--sim_seconds', type=int, default=None,
                   help='episode length override, for smoke tests (default: the network\'s 3600 s)')
    p.add_argument('--stop_after', type=int, default=None,
                   help='stop after this many episodes as an interrupted run would, state saved '
                        '(arlgos/check_arlgos.py uses it to test --resume)')
    return p.parse_args(argv)


def atomic_save(obj, path):
    import torch
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)          # a crash mid-write leaves the previous file intact


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def keep_rows_before(path, episode):
    """Drop log rows with episode >= `episode` (a resumed run plays them again)."""
    if not os.path.exists(path):
        return
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if int(r['episode']) < episode]
    os.remove(path)
    for r in rows:
        append_row(path, r)


def main(argv=None):
    args = parse_args(argv)
    os.environ['UNICORN_NET'] = args.net          # parameters.py reads it when first imported
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, 'arlgos'))
    import numpy as np
    import torch
    from parameters import SUMO_PARAMS
    from env.arlgos_env import ArlgosEnv
    from env.sumo_api import traci
    from models.Arlgos import ArlgosAgent
    from runner_arlgos import run_episode

    torch.set_num_threads(1)                      # tiny networks; parallel seeds must not fight
    name = args.name or 'arlgos_{}_seed{}'.format(args.net, args.seed)
    run_dir = os.path.join(REPO, 'Train_MATSC', name)
    model_dir = os.path.join(run_dir, 'model')
    latest = os.path.join(model_dir, 'latest.pt')
    train_csv, val_csv = os.path.join(run_dir, 'train.csv'), os.path.join(run_dir, 'val.csv')
    if args.resume:
        if not os.path.exists(latest):
            raise SystemExit('{} not found: nothing to resume'.format(latest))
        cfg_path = os.path.join(run_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                trained = json.load(f).get('lane_weight', 'max')
            if trained != args.lane_weight:
                raise SystemExit('{} trained with --lane_weight {}; resume it with the same value, not {}'.format(
                    name, trained, args.lane_weight))
    elif os.path.exists(run_dir):
        raise SystemExit('{} exists; pick another --name, or --resume'.format(run_dir))
    reserved = set(SUMO_PARAMS.TEST_SEEDS)
    clash = sorted(set(args.val_seeds) & reserved)
    if clash:
        raise SystemExit('--val_seeds {} are test seeds'.format(clash))
    os.makedirs(model_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)                     # the env draws its training SUMO seeds here
    server = args.server if args.server is not None else 100 + args.seed
    env = ArlgosEnv(server_number=server, test=False, reward=args.reward, lane_weight=args.lane_weight)
    if args.sim_seconds:
        env.max_sumo_step = args.sim_seconds
    spec = env.graph_spec()
    agent = ArlgosAgent(spec, episodes=args.episodes, seed=args.seed, algo=args.algo, conv=args.conv,
                        inter=not args.no_inter, agent_id=not args.no_agent_id, lr=args.lr,
                        gamma=args.gamma, batch_size=args.batch_size, memory_size=args.memory_size,
                        target_update=args.target_update, eps_decay=args.eps_decay, eps_min=args.eps_min,
                        pretrain_eps=args.pretrain_eps, hidden=args.hidden, mixer_embed=args.mixer_embed,
                        bootstrap_timeout=not args.terminal_at_timeout)

    start = 0
    if args.resume:
        state = torch.load(latest, map_location='cpu')
        agent.load_full(state['agent'])
        random.setstate(state['rng']['random'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        start = state['episode']
        keep_rows_before(train_csv, start)
        keep_rows_before(val_csv, start)
        print('[ARLGOS] resuming {} at episode {}'.format(name, start), flush=True)
    else:
        from common import machine_info
        cfg = dict(vars(args), name=name, server=server, method='ARLGOS-v2',
                   green_s=env.green_s, yellow_s=env.yellow_s, teleport_s=env.sumo_teleport_time,
                   episode_s=env.max_sumo_step, eps_min_used=agent.eps_min,
                   n_junctions=agent.n_agents, n_nodes=agent.n_nodes, max_phases=agent.max_phases,
                   flow_edges=int(spec['a_down'].sum()), n_parameters=agent.n_parameters(),
                   test_seeds=list(SUMO_PARAMS.TEST_SEEDS), machine=machine_info(),
                   sumo_client='libsumo' if os.environ.get('UNICORN_LIBSUMO') == '1' else 'traci',
                   demand=SUMO_PARAMS.DEMAND, fcd=SUMO_PARAMS.FCD)       # T15F: Fenglin's two switches
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=1, default=str)
    print('[ARLGOS] {}: {} junctions, {} lane nodes, {} flow edges, up to {} phases, {} parameters, '
          'eps floor {:.4f}, {} s per step'.format(name, agent.n_agents, agent.n_nodes,
                                                  int(spec['a_down'].sum()), agent.max_phases,
                                                  agent.n_parameters(), agent.eps_min, env.green_s), flush=True)

    def save_latest(next_episode):
        atomic_save({'agent': agent.full_state(), 'episode': next_episode,
                     'rng': {'random': random.getstate(), 'numpy': np.random.get_state(),
                             'torch': torch.get_rng_state()}}, latest)

    def validate(ep):
        t0 = time.time()
        queues = [run_episode(env, agent, ep, train=False, seed=s)['queue'] for s in args.val_seeds]
        mean = float(np.mean(queues))
        star = ''
        if mean < agent.best_val:
            agent.best_val = mean
            atomic_save(dict(agent.policy_state(), episode=ep, val_queue=mean),
                        os.path.join(model_dir, 'best_val.pt'))
            star = '  -> best_val.pt'
        row = {'episode': ep, 'val_queue': round(mean, 4)}
        row.update({'queue_{}'.format(s): round(q, 4) for s, q in zip(args.val_seeds, queues)})
        row['seconds'] = round(time.time() - t0, 1)
        append_row(val_csv, row)
        print('  [VAL] ep {}  queue {:.4f} over seeds {}{}'.format(ep, mean, args.val_seeds, star), flush=True)

    t_start, walls = time.time(), []
    try:
        for ep in range(start, args.episodes):
            out = run_episode(env, agent, ep, train=True)
            agent.end_episode(ep)
            walls.append(out['seconds'])
            row = {'episode': ep, 'sumo_seed': out['seed'], 'steps': out['steps'],
                   'reward': round(out['reward'], 4), 'queue': round(out['queue'], 4),
                   'loss': round(out['loss'], 6), 'epsilon': round(agent.epsilon, 4),
                   'switch_rate': round(out['switch_rate'], 4), 'buffer': agent.memory.size,
                   'seconds': out['seconds']}
            append_row(train_csv, row)
            left = (args.episodes - ep - 1) * (sum(walls[-20:]) / len(walls[-20:]))
            print('EP {:04d}/{}  seed {}  queue {:.3f}  reward {:.2f}  loss {:.4f}  eps {:.3f}  '
                  '{:.0f} s  (about {:.1f} h of training left)'.format(
                      ep, args.episodes, out['seed'], out['queue'], out['reward'], out['loss'],
                      agent.epsilon, out['seconds'], left / 3600.0), flush=True)
            last = ep == args.episodes - 1
            if args.val_seeds and ep >= args.pretrain_eps and ((ep + 1) % args.val_every == 0 or last):
                validate(ep)
            if (ep + 1) % args.save_every == 0 or last:
                atomic_save(dict(agent.policy_state(), episode=ep),
                            os.path.join(model_dir, 'ckpt_{:04d}.pt'.format(ep + 1)))
            if (ep + 1) % args.latest_every == 0 or last:
                save_latest(ep + 1)
            if args.stop_after and ep + 1 >= args.stop_after and not last:
                save_latest(ep + 1)
                print('[ARLGOS] stopped after {} episodes (--stop_after)'.format(ep + 1), flush=True)
                return
    finally:
        try:
            traci.close()
        except Exception:
            pass
    with open(os.path.join(run_dir, 'done.json'), 'w') as f:
        json.dump({'episodes': args.episodes, 'hours': round((time.time() - t_start) / 3600.0, 2),
                   'best_val_queue': agent.best_val}, f)
    print('[ARLGOS] done: {}'.format(run_dir), flush=True)


if __name__ == '__main__':
    main()
