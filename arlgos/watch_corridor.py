"""ARLGOS: watch a Unicorn corridor checkpoint drive the corridor in sumo-gui. For looking only.

    python arlgos/watch_corridor.py [--run Train_MATSC/corridor3_full_seed1]
                                    [--checkpoint auto|best|latest|<N>] [--seed 9042]
                                    [--delay_ms 50] [--decisions 90]
    (ARLGOS's watch_sumo.ps1 calls this with the unicorn env's Python.)

One greedy episode, as eval_corridor.py's 'greedy' mode plays it: the 300 s fixed-time
warm-up runs at full speed, then the 90 decisions (900 s) at --delay_ms per 0.05 s step
(50 = real time; sumo-gui's "Delay" box slows it further). The console prints every
decision: each junction's green, whether Unicorn kept or switched it, and the queue.

Which checkpoint (--checkpoint):
  auto   : 'best' once the run has been evaluated (eval/best.json), else 'latest'
  best   : the checkpoint that validation chose (the one that was tested)
  latest : the newest numbered snapshot checkpoint{N}.pkl (saved every 50 updates)
  <N>    : checkpoint<N>.pkl
It never opens model/checkpoint.pkl, the resume file training replaces every update: on
Windows, a reader holding that file open makes training's os.replace fail.

Nothing is written or scored. The default seed 9042 lies outside every reserved range
(test 1042-1161, validation 5042-5061) and ARLGOS's training ranges; one demo seed says
nothing about performance. sumo-gui needs traci, so UNICORN_LIBSUMO is cleared here.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
DEMO_SEED = 9042
SETTLE_S = 10                        # a numbered snapshot this new may still be being written


def pick_run(run):
    """The --run folder, or the corridor run whose training state changed last."""
    if run:
        return os.path.abspath(run if os.path.isabs(run) else os.path.join(FORK, run))
    marks = glob.glob(os.path.join(FORK, 'Train_MATSC', 'corridor3_*', 'model', 'checkpoint*.pkl'))
    if not marks:
        raise SystemExit('no Train_MATSC/corridor3_* run has a checkpoint yet')
    return os.path.dirname(os.path.dirname(max(marks, key=os.path.getmtime)))


def numbered(model_dir):
    """checkpoint{N}.pkl files as [(N, path)], oldest first; never checkpoint.pkl."""
    out = []
    for p in glob.glob(os.path.join(model_dir, 'checkpoint*.pkl')):
        m = re.search(r'checkpoint(\d+)\.pkl$', p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def pick_checkpoint(run, which='auto', now=None):
    """(path, label) of the checkpoint to watch; see the module docstring."""
    model_dir = os.path.join(run, 'model')
    best_json = os.path.join(run, 'eval', 'best.json')
    if which == 'auto':
        which = 'best' if os.path.exists(best_json) else 'latest'
    if which == 'best':
        if not os.path.exists(best_json):
            raise SystemExit('{} has not been evaluated yet (no eval/best.json); '
                             'use -Checkpoint latest'.format(os.path.basename(run)))
        with open(best_json) as f:
            name = json.load(f)['checkpoint']
        return os.path.join(model_dir, name), 'best on validation ({})'.format(name)
    snaps = numbered(model_dir)
    if which == 'latest':
        now = time.time() if now is None else now
        settled = [s for s in snaps if now - os.path.getmtime(s[1]) > SETTLE_S]
        if not settled:
            raise SystemExit('{}: no snapshot yet (the first one comes at update 50)'.format(
                os.path.basename(run)))
        n, path = settled[-1]
        return path, 'latest snapshot (update {})'.format(n)
    n = int(which)
    path = os.path.join(model_dir, 'checkpoint{}.pkl'.format(n))
    if not os.path.exists(path):
        raise SystemExit('no {}; snapshots: {}'.format(path, [s[0] for s in snaps]))
    return path, 'update {}'.format(n)


def gui_env_class(CorridorEnv, sumo_binary, traci, delay_ms):
    """CorridorEnv with sumo-gui: the same scenario arguments plus --start and --delay 0, the
    warm-up at full speed, then delay_ms of wall clock per step. SUMO 1.28's TraCI has no
    gui.setDelay, so this client paces the steps (as ARLGOS's SumoEnv does). Nothing else differs."""
    class GuiCorridorEnv(CorridorEnv):
        _pace = None                                  # [wall-clock start, steps] in the episode

        def sumo_args(self, seed):
            return super(GuiCorridorEnv, self).sumo_args(seed) + ['--start', '--quit-on-end',
                                                                  '--delay', '0']

        def set_world(self, seed, write_unfinished=False):
            args = self.sumo_args(seed)
            if self._started:
                traci.load(args)
            else:
                traci.start([sumo_binary('sumo-gui')] + args)
                self._started = True
            return 0

        def start_episode(self, seed):
            self._pace = None                         # the warm-up at full speed
            obs = super(GuiCorridorEnv, self).start_episode(seed)
            if delay_ms > 0:
                self._pace = [time.perf_counter(), 0]
            return obs

        def tick(self):
            super(GuiCorridorEnv, self).tick()
            if self._pace is not None:
                self._pace[1] += 1
                lag = self._pace[0] + self._pace[1] * delay_ms / 1000.0 - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
    return GuiCorridorEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='', help='Train_MATSC/<name> (default: the one training now)')
    ap.add_argument('--checkpoint', default='auto', help='auto | best | latest | <N>')
    ap.add_argument('--seed', type=int, default=DEMO_SEED)
    ap.add_argument('--delay_ms', type=int, default=50, help='per 0.05 s step; 50 = real time')
    ap.add_argument('--decisions', type=int, default=90, help='stop after this many (90 = 900 s)')
    args = ap.parse_args()

    os.environ.pop('UNICORN_LIBSUMO', None)          # before env.sumo_api is imported
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    sys.path.insert(0, HERE)
    run = pick_run(args.run)
    from eval_corridor import read_run_config
    sensing = read_run_config(os.path.join(run, 'config.json')).get('SENSING', 'full')
    os.environ['UNICORN_NET'] = 'corridor_3'
    os.environ['UNICORN_SENSING'] = sensing
    ckpt, label = pick_checkpoint(run, args.checkpoint)

    import numpy as np
    import torch
    torch.set_num_threads(1)                         # leave the cores to training
    from common import use_env_sumo
    use_env_sumo()
    from env.corridor import CorridorEnv, sumo_binary, PHASE_OF_ACTION, traci
    from env.corridor_signals import EW_GREEN, NS_GREEN
    from models.Unicorn import Unicorn

    env = gui_env_class(CorridorEnv, sumo_binary, traci, args.delay_ms)(None, test=True, sensing=sensing)
    model = Unicorn(input_dim=env.tls_obs_space, agent_dim=env.tls_agent_space,
                    int_vec_dim=env.tls_int_attr_space, actor_lr=1e-4, critic_lr=1e-4)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print('Run        : {}  ({} sensing)'.format(os.path.basename(run), sensing))
    print('Checkpoint : {}'.format(label))
    print('Seed       : {} (demo only, never scored)'.format(args.seed))
    print('sumo-gui   : warm-up 300 s at full speed, then {} ms per 0.05 s step{}'.format(
        args.delay_ms, ' (real time)' if args.delay_ms == 50 else ''))
    print('Each line  : time | mean queue per camera now | per junction: its green, and what Unicorn '
          'asks (keep to 90 s, or SWITCH = end it at 30 s)\n')

    short = {EW_GREEN: 'EW', NS_GREEN: 'NS'}

    def tensor(d):
        return torch.tensor(np.array(list(d.values())), dtype=torch.float32)
    try:
        torch.manual_seed(21 + args.seed)            # the VAE noise, seeded as eval_corridor.py does
        obs = env.reset_test(0, args.seed)
        h, done = None, False
        while not done and env.rl_step < args.decisions:
            phase_vec, mask = env.get_phase_vec_mask_dict()
            int_attr = env.get_int_attr_vec_dict()
            with torch.no_grad():
                policy, h, _, _, _ = model(tensor(obs), tensor(phase_vec), tensor(int_attr), tensor(mask), h)
            action = policy.argmax(-1).reshape(-1)
            acts = {tls: int(action[i]) for i, tls in enumerate(env.rl_tls_list)}
            parts = []
            for ctrl, tls in zip(env.signals, env.tls_list):
                want = PHASE_OF_ACTION[acts[tls]]
                if ctrl.phase in short:
                    what = 'keep' if want == ctrl.phase else 'SWITCH'
                    parts.append('{} {} {:<6}'.format(tls, short[ctrl.phase], what))
                else:
                    parts.append('{} {:<9}'.format(tls, '(change)'))
            print('{:>4d} s | queue {:5.2f} | {}'.format(env.rl_step * 10, env.aql_series[-1], ' | '.join(parts)))
            obs, _, done, _ = env.step(acts)
        print('\nMean queue per camera over this episode: {:.3f} (one demo seed; '
              'results come from eval_corridor.py)'.format(env.aql()))
    except traci.exceptions.FatalTraCIError:
        print('\nsumo-gui was closed.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
