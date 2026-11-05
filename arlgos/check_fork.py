"""ARLGOS: self-checks for the fork's changes (run after every change to the fork).

    python arlgos/check_fork.py                         # env unicorn, expects SUMO 1.28.0
    python arlgos/check_fork.py --expect-sumo 1.16.0    # env unicorn_sumo116

Checks that
  1. parameters.py has the paper's Sec. V-C training values, and GAE lambda is
     the value the published code hard-codes (0.95, see UNICORN_SETUP.md);
  2. the defaults are Cologne8 on CPU, 1500 episodes, checkpoints every 50;
  3. each dataset family gets the README's green / yellow / teleport times;
  4. TEST_SEEDS are the evaluators' seeds, and draw_train_seed() never returns
     a test or validation seed;
  5. the environment builds, and a Cologne8 simulation starts with the
     expected SUMO, whose TraCI API version matches the traci client;
  6. resuming (T4): UNICORN_RESUME switches LOAD_MODEL on for the named run,
     save_latest / load_latest restore weights, optimisers, the episode counter
     and the RNG streams, and the driver and workers use them.
Exits 1 if any check fails.
"""
import argparse
import os
import re
import subprocess
import sys

from common import REPO, sumo_binary_version, use_env_sumo

FAILED = []


def check(name, ok, detail=''):
    print('[{}] {}{}'.format('PASS' if ok else 'FAIL', name, ': ' + detail if detail else ''))
    if not ok:
        FAILED.append(name)


def timing_for(net):
    """GREEN/YELLOW/TELEPORT that parameters.py gives `net`, imported fresh with UNICORN_NET set."""
    env = dict(os.environ, UNICORN_NET=net)
    code = 'from parameters import SUMO_PARAMS as P; print(P.GREEN_DURATION, P.YELLOW_DURATION, P.TELEPORT_TIME)'
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env, capture_output=True, text=True)
    return tuple(int(x) for x in out.stdout.split()) if out.returncode == 0 else out.stderr.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-sumo', default='1.28.0')
    args = parser.parse_args()
    for var in ('UNICORN_NET', 'UNICORN_MAX_EPISODES', 'UNICORN_WORKERS', 'UNICORN_SEED', 'UNICORN_EXP_NAME',
                'UNICORN_RESUME'):
        os.environ.pop(var, None)                 # check the file's defaults
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import numpy as np
    from parameters import INPUT_PARAMS, NETWORK_PARAMS, SUMO_PARAMS, TRAIN_PARAMS

    # 1. paper Sec. V-C
    paper = {'GAMMA': 0.95, 'A_LR_Q': 1e-4, 'C_LR_Q': 2e-4, 'VL_FACTOR': 0.5, 'EL_FACTOR': 2e-3,
             'PL_FACTOR': 2e-4, 'CL_FACTOR': 1e-5, 'EPS_CLIP': 0.2, 'K_EPOCH': 6,
             'CONTRASTIVE_TEMP': 0.2, 'CONTRASTIVE_BATCH': 256}
    wrong = {k: getattr(NETWORK_PARAMS, k) for k, v in paper.items() if getattr(NETWORK_PARAMS, k) != v}
    check('paper Sec. V-C values in NETWORK_PARAMS', not wrong, str(wrong) if wrong else '')
    with open(os.path.join(REPO, 'runner_unicorn.py')) as f:
        runner = f.read()
    check('GAE lambda = 0.95 as published (runner_unicorn.py hard-codes gamma * 0.95)',
          'calculate_discount_return(deltas, gamma * 0.95)' in runner)

    # 2. defaults
    defaults = {'NET_NAME': (SUMO_PARAMS.NET_NAME, 'cologne_network_8'),
                'MAX_EPISODES': (INPUT_PARAMS.MAX_EPISODES, 1500),
                'NUM_META_AGENTS': (INPUT_PARAMS.NUM_META_AGENTS, 6),
                'USE_GPU': (TRAIN_PARAMS.USE_GPU, False), 'NUM_GPU': (TRAIN_PARAMS.NUM_GPU, 0),
                'SAVE_MODEL_STEP': (TRAIN_PARAMS.SAVE_MODEL_STEP, 50), 'TRAIN_SEED': (TRAIN_PARAMS.TRAIN_SEED, 0),
                'RANDOM_SEED': (SUMO_PARAMS.RANDOM_SEED, True)}
    wrong = {k: got for k, (got, want) in defaults.items() if got != want}
    check('defaults: Cologne8, 1500 episodes, 6 workers, CPU, checkpoints every 50', not wrong,
          str(wrong) if wrong else '')

    # 3. timing per dataset family (README "Recommended settings", paper Sec. V-C)
    expected = {'cologne_network_8': (15, 5, -1), 'arterial_network_4_4': (15, 5, -1),
                'grid_network_5_5': (10, 3, 300), 'monaco_network_30': (10, 3, 300),
                'shaoxing_network_7': (15, 5, 600)}
    got = {net: timing_for(net) for net in expected}
    wrong = {net: g for net, g in got.items() if g != expected[net]}
    check('green / yellow / teleport per dataset family', not wrong, str(wrong) if wrong else '')

    # 4. seeds
    check('TEST_SEEDS = 1000 + 100 * i, i < 10', SUMO_PARAMS.TEST_SEEDS == [1000 + 100 * i for i in range(10)])
    for evaluator in ('evaluator_non_rl.py', 'evaluator_rl.py'):
        with open(os.path.join(REPO, evaluator)) as f:
            src = f.read()
        values = {k: re.search(r'^\s*{}\s*=\s*(\d+)'.format(k), src, re.M) for k in ('seed', 'test_num', 'start_seed')}
        values = {k: int(m.group(1)) if m else None for k, m in values.items()}
        # seeds = [start_seed + seed * i ...] (evaluator_non_rl.py) or [1000 + seed * i ...] (evaluator_rl.py)
        start = re.search(r'seeds\s*=\s*\[\s*(\w+)\s*\+\s*seed\s*\*\s*i\s+for\s+i\s+in\s+range\(test_num\)', src)
        start = start and (int(start.group(1)) if start.group(1).isdigit() else values.get(start.group(1)))
        seeds = ([start + values['seed'] * i for i in range(values['test_num'])]
                 if start is not None and values['seed'] is not None and values['test_num'] is not None else None)
        check('{} tests on TEST_SEEDS'.format(evaluator), seeds == SUMO_PARAMS.TEST_SEEDS, str(values))
    from env import matsc
    queue = [1000, 5042, 1900, 5061, 7]
    real_randint = np.random.randint
    np.random.randint = lambda low, high: queue.pop(0)
    try:
        first = matsc.draw_train_seed()
    finally:
        np.random.randint = real_randint
    check('draw_train_seed() skips reserved seeds (mocked draws 1000, 5042, 1900, 5061, 7)', first == 7, str(first))
    np.random.seed(0)
    reserved = set(SUMO_PARAMS.TEST_SEEDS) | set(SUMO_PARAMS.VAL_SEEDS)
    hits = sum(matsc.draw_train_seed() in reserved for _ in range(20000))
    check('20000 draws, none reserved', hits == 0, '{} hits'.format(hits))

    # 5. environment and SUMO
    use_env_sumo()
    from utils import set_env
    env = set_env(server_number=None)
    check('MATSC_Env builds for Cologne8', len(env.tls_list) == 8, '{} intersections'.format(len(env.tls_list)))
    version, binary = sumo_binary_version()
    check('sumo binary is SUMO {}'.format(args.expect_sumo), version == args.expect_sumo, binary)
    import traci
    cfg = os.path.join('maps', 'cologne_network_8', 'cologne_network_8.sumocfg')
    traci.start([binary, '-c', cfg, '--no-step-log', 'true', '--no-warnings', 'true'], port=6099)
    try:
        traci.simulationStep()
        api, reported = traci.getVersion()
    finally:
        traci.close()
    check('Cologne8 simulation starts; SUMO reports {}'.format(args.expect_sumo), args.expect_sumo in reported, reported)
    check('TraCI API: server {} = client {}'.format(api, traci.constants.TRACI_VERSION),
          api == traci.constants.TRACI_VERSION)

    # 6. resuming (T4)
    code = 'from parameters import TRAIN_PARAMS as T; print(T.LOAD_MODEL, T.EXPERIMENT_PATH)'
    runs = {}
    for resume in ('', '1'):
        env = dict(os.environ, UNICORN_RESUME=resume, UNICORN_EXP_NAME='some_run')
        out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env, capture_output=True, text=True)
        runs[resume] = out.stdout.strip() or out.stderr.strip()
    check('UNICORN_RESUME=1 resumes the named run, unset starts a new one',
          runs == {'': 'False None', '1': 'True ./Train_MATSC/some_run'}, str(runs))

    import random
    import tempfile
    import torch
    from models.Unicorn import Unicorn
    from utils import save_latest, load_latest

    def model():
        torch.manual_seed(0)
        return Unicorn(input_dim=[22, 5], agent_dim=5, int_vec_dim=56, actor_lr=1e-4, critic_lr=2e-4)

    net = model()
    s, pv, iv = torch.randn((4, 5, 22, 5)), torch.randn((4, 5, 6, 22)), torch.randn((4, 5, 6, 56))
    mask, na = torch.zeros((4, 5, 6)), torch.zeros((4, 5, 22))
    pi, _, _, _, _ = net.forward(s, pv, iv, mask, None)
    v, _, _, _, _ = net.forward_v(s, pv, iv, mask, None, na)
    (pi.sum() + v.sum()).backward()
    net.actor_optimizer.step()
    net.critic_optimizer.step()
    with tempfile.TemporaryDirectory() as d:
        save_latest(net, 37, d)
        after_save = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        other = model()
        epoch = load_latest(other, d)
        after_load = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        leftovers = [f for f in os.listdir(d) if f != 'checkpoint.pkl']
    same_w = all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), other.state_dict().values()))
    adam = [o.state_dict()['state'] for o in (net.actor_optimizer, other.actor_optimizer)]
    same_opt = bool(adam[0]) and all(torch.equal(adam[0][k]['exp_avg'], adam[1][k]['exp_avg']) for k in adam[0])
    check('save_latest -> load_latest: weights, Adam state, counter 37, random/numpy/torch streams',
          same_w and same_opt and epoch == 37 and after_save == after_load and not leftovers,
          'weights {} adam {} epoch {} rng {} files {}'.format(same_w, same_opt, epoch,
                                                               after_save == after_load, leftovers))
    with open(os.path.join(REPO, 'driver_unicorn.py')) as f:
        driver = f.read()
    with open(os.path.join(REPO, 'runner_unicorn.py')) as f:
        runner = f.read()
    check('driver saves the latest state every update and resumes from it; workers get the start episode',
          'save_latest(global_network, curr_episode' in driver and 'load_latest(global_network' in driver
          and 'Runner.remote(i, start_episode)' in driver
          and 'def __init__(self, meta_agent_id, start_episode=0)' in runner)
    check("workers' resume path has its separator (upstream: EXPERIMENT_PATH + 'model/...')",
          "EXPERIMENT_PATH + 'model/" not in runner
          and "os.path.join(TRAIN_PARAMS.EXPERIMENT_PATH, 'model', 'state_dict.pth')" in runner)

    print('\n{} failed'.format(len(FAILED)) if FAILED else '\nall checks passed')
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
