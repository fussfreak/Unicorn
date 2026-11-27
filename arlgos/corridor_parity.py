"""ARLGOS (T9): does the fork's corridor env simulate exactly what ARLGOS's SumoEnv does?

Three steps, two Python environments (docs/T9_design.md section 3, checks 1-4):

    <unicorn python>     arlgos/corridor_parity.py fork   --out results/T9/parity_fork.json
    <arlgos_sumo python> arlgos/corridor_parity.py arlgos --out results/T9/parity_arlgos.json
    <arlgos_sumo python> arlgos/corridor_parity.py compare                 (writes parity.md)

arlgos/parity.ps1 runs all three. Each side runs the same episodes:
  * 'fixed60'   : no actions, 60 s greens (ARLGOS's fixed-time baseline);
  * 'keepswitch': at every decision each junction keeps or switches its green by a
                  fixed pseudo-random sequence (seeded per demand seed, independent of
                  the state), so both sides get the same commands. The fork side sends
                  them as Unicorn phase indices through CorridorEnv.step; the ARLGOS side
                  as green targets (switch -> 30 s, keep or transition -> 90 s), as
                  pressure.max_pressure_greens does.
Per episode both record: every departure (time, vehicle id), the three light strings
after every 0.05 s tick, the camera counts at t = 0 and the AQL series; and with
--outputs, SUMO's T7 files, which 'compare' scores with ARLGOS's sumo_metrics.py.
The fork side also checks that its observe_full equals Unicorn's own
Tls.observe_resco_network at every decision where a junction shows a green.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
DEFAULT_ARLGOS = os.path.join(os.path.dirname(FORK), 'ARLGOS')
OUT = os.path.join(FORK, 'results', 'T9')
SEEDS = [1042, 1043, 1044]
POLICIES = ('fixed60', 'keepswitch')
TLS = ('C3', 'C2', 'C1')
N_DECISIONS, DECISION_STEPS, SWITCH_P = 90, 200, 0.3


def switch_bits(seed):
    import numpy as np
    return np.random.RandomState(seed + 777).rand(N_DECISIONS, 3) < SWITCH_P


class Trace:
    def __init__(self, getstate, departed):
        self.getstate, self.departed = getstate, departed
        self.lights, self.deps, self.n = hashlib.sha1(), [], 0

    def after_tick(self, t_now):
        self.lights.update('|'.join(self.getstate(t) for t in TLS).encode())
        self.deps.extend((round(t_now, 2), v) for v in self.departed())
        self.n += 1

    def result(self):
        h = hashlib.sha1(json.dumps(self.deps).encode()).hexdigest()
        return {'ticks': self.n, 'lights_sha1': self.lights.hexdigest(), 'departures': len(self.deps),
                'departures_sha1': h, 'first_departures': self.deps[:50]}


# -- fork side (env unicorn) ------------------------------------------------------------------
def run_fork(args):
    os.chdir(FORK)
    sys.path.insert(0, FORK)
    sys.path.insert(0, HERE)
    from common import use_env_sumo
    use_env_sumo()
    import numpy as np
    from env.sumo_api import traci            # the client the env uses (UNICORN_LIBSUMO)
    from env.corridor import CorridorEnv, PHASE_OF_ACTION
    from env.corridor_signals import GREEN_PHASES
    out = {'side': 'fork', 'client': traci.__name__, 'python': sys.version.split()[0], 'episodes': {}}
    obs_checked = obs_mismatch = 0
    for policy in POLICIES:
        outputs = os.path.join(args.outputs, 'fork', policy) if args.outputs else None
        env = CorridorEnv(None, test=True, sensing='full', outputs=outputs)
        orig_tick = env.tick
        for seed in args.seeds:
            tr = Trace(traci.trafficlight.getRedYellowGreenState, traci.simulation.getDepartedIDList)

            def tick():
                orig_tick()
                tr.after_tick(traci.simulation.getTime())
            env.tick = tick
            env.start_episode(seed)
            counts0 = env.camera_counts()
            bits = switch_bits(seed)
            for k in range(N_DECISIONS):
                if policy == 'fixed60':
                    for _ in range(DECISION_STEPS):
                        env.tick()
                    env.rl_step += 1
                    if env.rl_step < N_DECISIONS:
                        env.record_counts()
                    continue
                for tid in env.tls_list:
                    # SUMO shows a new phase one tick after the controller enters it (ARLGOS
                    # writes the lights before each step), so test what SUMO shows
                    if traci.trafficlight.getRedYellowGreenState(tid) in env.tls_dict[tid].action_space:
                        a = env.observe_full(tid)
                        b = env.tls_dict[tid].observe_resco_network(env.outgoing_lane_state_dict)
                        obs_checked += 1
                        obs_mismatch += int(not np.array_equal(a, b))
                actions = dict(env.forced_actions())
                for j, (ctrl, tid) in enumerate(zip(env.signals, env.tls_list)):
                    if tid not in actions:
                        cur = PHASE_OF_ACTION.index(ctrl.phase)
                        actions[tid] = 1 - cur if bits[k][j] else cur
                env.step(actions)
            out['episodes']['{}/{}'.format(policy, seed)] = dict(
                tr.result(), counts_t0=counts0, aql_series=env.aql_series[:N_DECISIONS], aql=env.aql())
            print('fork {} {}: AQL {:.4f}, {} departures'.format(policy, seed, env.aql(), len(tr.deps)))
        env.close()
    out['observe_full_checked'], out['observe_full_mismatch'] = obs_checked, obs_mismatch
    write(args.out, out)


# -- ARLGOS side (env arlgos_sumo) ------------------------------------------------------------
def run_arlgos(args):
    sys.path.insert(0, args.arlgos)
    os.environ['PYTHONPATH'] = ''
    import numpy as np
    from sumo_env import SumoEnv
    from signal_control import CorridorSignals, MIN_GREEN, MAX_GREEN, GREEN_PHASES
    out = {'side': 'arlgos', 'episodes': {}}
    for policy in POLICIES:
        outputs = os.path.join(args.outputs, 'arlgos', policy) if args.outputs else None
        signals = CorridorSignals()
        env = SumoEnv(signals, outputs=outputs)
        orig_tick = env.tick
        for seed in args.seeds:
            tr = Trace(lambda t: env.t.trafficlight.getRedYellowGreenState(t),
                       lambda: env.t.simulation.getDepartedIDList())

            def tick(phases, sim_time):
                orig_tick(phases, sim_time)
                tr.after_tick(env.t.simulation.getTime())
            env.tick = tick
            env.reset(seed)
            counts0 = env.detect()[1]
            bits = switch_bits(seed)
            aqls = []
            for k in range(N_DECISIONS):
                counts = env.detect()[1]
                aqls.append(float(np.mean(counts)))
                if policy == 'keepswitch':
                    signals.set_targets([MIN_GREEN if (bits[k][j] and c.phase in GREEN_PHASES) else MAX_GREEN
                                         for j, c in enumerate(signals)])
                for _ in range(DECISION_STEPS):
                    env.tick(None, 0.0)
            out['episodes']['{}/{}'.format(policy, seed)] = dict(
                tr.result(), counts_t0=counts0, aql_series=aqls, aql=float(np.mean(aqls)))
            print('arlgos {} {}: AQL {:.4f}, {} departures'.format(policy, seed, float(np.mean(aqls)),
                                                                  len(tr.deps)))
        env.close()
    out['scenario_fingerprint'] = __import__('sumo_env').scenario_fingerprint()
    write(args.out, out)


# -- compare (env arlgos_sumo: needs ARLGOS's sumo_metrics.py) -------------------------------
def compare(args):
    with open(os.path.join(OUT, 'parity_fork.json')) as f:
        fork = json.load(f)
    with open(os.path.join(OUT, 'parity_arlgos.json')) as f:
        arl = json.load(f)
    rows, failed = [], []
    for key in sorted(arl['episodes']):
        a, b = arl['episodes'][key], fork['episodes'].get(key)
        if b is None:
            failed.append(key + ' missing on the fork side')
            continue
        same = {
            'demand (departure times and ids)': a['departures_sha1'] == b['departures_sha1'],
            'light strings, every tick': a['lights_sha1'] == b['lights_sha1'] and a['ticks'] == b['ticks'],
            'camera counts at t = 0': a['counts_t0'] == b['counts_t0'],
            'AQL series': a['aql_series'] == b['aql_series'],
        }
        if args.outputs:
            sys.path.insert(0, args.arlgos)
            from sumo_metrics import episode_metrics
            policy, seed = key.split('/')
            ma = episode_metrics(os.path.join(args.outputs, 'arlgos', policy, seed))
            mb = episode_metrics(os.path.join(args.outputs, 'fork', policy, seed))
            diff = max(abs(float(ma[k]) - float(mb[k])) for k in ma if isinstance(ma[k], (int, float)))
            same['T7 metrics (max abs diff {:.2g})'.format(diff)] = diff == 0
        for what, ok in same.items():
            rows.append('| {} | {} | {} |'.format(key, what, 'same' if ok else '**DIFFERENT**'))
            if not ok:
                failed.append('{}: {}'.format(key, what))
    obs_ok = fork.get('observe_full_mismatch', 1) == 0 and fork.get('observe_full_checked', 0) > 0
    if not obs_ok:
        failed.append('observe_full vs observe_resco_network')
    client = fork.get('client', 'traci')
    md = ['# T9 parity: the fork\'s corridor env vs ARLGOS\'s SumoEnv', '',
          'Seeds {}; policies fixed-time 60 s and a fixed keep/switch sequence (switch probability {}). '
          'Fork: {} (Python {}). ARLGOS: libsumo (env arlgos_sumo). Scenario fingerprint {}.'.format(
              ', '.join(str(s) for s in SEEDS), SWITCH_P, client, fork.get('python', '3.8'),
              arl.get('scenario_fingerprint')), '',
          '| Episode | What | Result |', '|---|---|---|'] + rows + [
          '', 'observe_full equals Unicorn\'s Tls.observe_resco_network at {} junction-decisions in green: '
          '{} mismatches.'.format(fork.get('observe_full_checked'), fork.get('observe_full_mismatch')), '',
          '**{}**'.format('PASS: identical' if not failed else 'FAIL: ' + '; '.join(failed))]
    for name in ('parity.md', 'parity_{}.md'.format(client)):    # the latest, and one per client
        with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    sys.exit(1 if failed else 0)


def write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('side', choices=('fork', 'arlgos', 'compare'))
    ap.add_argument('--arlgos', default=DEFAULT_ARLGOS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--outputs', default=os.path.join(OUT, 'parity_outputs'),
                    help='where both sides write SUMO\'s T7 files ("" = none)')
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(OUT, 'parity_{}.json'.format(args.side))
    {'fork': run_fork, 'arlgos': run_arlgos, 'compare': compare}[args.side](args)


if __name__ == '__main__':
    main()
