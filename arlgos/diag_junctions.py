"""ARLGOS (T15, 2026-11-23): per-junction diagnosis of ARLGOS-v2 against Greedy.

The first step of Gate G4's failure branch (ROADMAP §4: "check the graph builder and phase head
first"), approved by the user on 2026-11-23.

    python arlgos/diag_junctions.py --run Train_MATSC/arlgos_cologne_network_8_seed1
    python arlgos/diag_junctions.py --selftest        # the static checks only, no simulation

Three parts, written to results/T15/diag_<net>.{md,json}:

1. Phase audit, static (each network's <net>_config.json, no simulation). ArlgosEnv describes a
   phase by the LANES it gives green, each lane taking its best movement (G = 1, g = 0.5, r = 0;
   env/arlgos_env.py _build_static). Two phases that differ only in which movements of the same
   lanes are green (for example "through + permitted left" and "protected left only" on a
   shared lane) then get identical lane weights: they are ALIASED. For such a pair the model's
   phase summary e_{j,p}, the node feature "signal" and the feature "red_t" are identical, so
   Q_j(p) and Q_j(q) differ only through the "is p showing now" bit. The audit lists every
   aliased pair, which of the two serves a strict subset of the other's movements, and does so
   for every network in maps/.

2. Phase audit, live (one SUMO start): SUMO's controlled links are in the order
   traffic_movement_list assumes; every controlled incoming lane is a graph node; the padded
   phases of the model are zero and masked; at every decision the light string SUMO shows equals
   action_space[the phase ArlgosEnv believes is showing].

3. Replay on the VALIDATION seeds (5042-5046; the test seeds are never touched) of the chosen
   checkpoint (greedy) and of Unicorn's Greedy controller (Tls.get_greedy_action, the call
   evaluator_non_rl.py makes), both in the same ArlgosEnv, so the traffic rules are identical.
   Every simulated second it records, per junction, read-only:
     * the halting vehicles on the junction's incoming lanes (whole lane, the evaluator's queue);
       the network queue is their sum over junctions / all lanes, so the gap splits exactly;
     * the waiting time (traci.vehicle.getWaitingTime, the evaluator's "intersection delay") of
       every vehicle in the network, credited to the junction whose incoming lane it is on, else
       to "elsewhere"; the network value is the sum / vehicles, so it splits exactly too;
     * the phase being served, for the share of time in the subset phase of an aliased pair.
   The decision-point queue it reports must equal the validation score the training run logged
   for that checkpoint (and T14's 0.210 for Greedy): that confirms the replay is the real one.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VAL_SEEDS = (5042, 5043, 5044, 5045, 5046)
LONG_WAIT_S = 60.0


# -- 1. static phase audit ---------------------------------------------------------------
def code_weight(ch):
    """ArlgosEnv's convention (env/arlgos_env.py code_weight)."""
    return 1.0 if ch == 'G' else 0.5 if ch == 'g' else 0.0


def green_set(code):
    return frozenset(i for i, ch in enumerate(code) if ch in 'Gg')


def lane_weights(junction, rule='max'):
    """(lanes, movements, W) with W[p, l] = the weight of lane l under phase p: its best
    movement ('max', ArlgosEnv's default) or the average of its movements ('mean'). Written
    independently of ArlgosEnv._build_static, so the live check can compare the two."""
    lanes = list(junction['incoming_lane_list'])
    moves = [(i, o) for i, outs in junction['lane_links'].items() for o in outs]
    phases = junction['valid_phase_list']
    w = np.zeros((len(phases), len(lanes)), dtype=np.float32)
    for p, code in enumerate(phases):
        per_lane = [[] for _ in lanes]
        for ch, (in_lane, _) in zip(code, moves):
            per_lane[lanes.index(in_lane)].append(code_weight(ch))
        for li, vals in enumerate(per_lane):
            if vals:
                w[p, li] = max(vals) if rule == 'max' else sum(vals) / len(vals)
    return lanes, moves, w


def phase_audit(config, rule='max'):
    """Per junction: phases, lanes, whether action_space is valid_phase_list, and the aliased
    phase pairs under the lane-weight rule ({'phases': [p, q], 'subset': the phase serving a
    strict subset of the other's movements, or None})."""
    out = OrderedDict()
    for tls_id, junction in config.items():
        lanes, moves, w = lane_weights(junction, rule)
        phases = junction['valid_phase_list']
        pairs = []
        for p in range(len(phases)):
            for q in range(p + 1, len(phases)):
                if np.allclose(w[p], w[q]):
                    gp, gq = green_set(phases[p]), green_set(phases[q])
                    pairs.append({'phases': [p, q], 'subset': q if gq < gp else p if gp < gq else None})
        out[tls_id] = {'n_phases': len(phases), 'n_lanes': len(lanes), 'n_movements': len(moves),
                       'action_space_is_valid_phase_list': junction['action_space'] == phases,
                       'phases': list(phases), 'lane_weights': w.tolist(), 'aliased': pairs}
    return out


def load_config(net):
    with open(os.path.join(REPO, 'maps', net, '{}_config.json'.format(net))) as f:
        return json.load(f)


def audit_all_networks():
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, 'maps', '*', '*_config.json'))):
        net = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                config = json.load(f)
            audit = phase_audit(config)
            audit_mean = phase_audit(config, 'mean')
        except (ValueError, KeyError, TypeError, AttributeError):
            continue                                   # not a Unicorn junction config
        rows.append({'net': net, 'junctions': len(audit),
                     'phases': sum(a['n_phases'] for a in audit.values()),
                     'aliased_pairs': sum(len(a['aliased']) for a in audit.values()),
                     'aliased_junctions': sum(1 for a in audit.values() if a['aliased']),
                     'aliased_pairs_mean': sum(len(a['aliased']) for a in audit_mean.values()),
                     'action_space_differs': sum(1 for a in audit.values() if not a['action_space_is_valid_phase_list'])})
    return rows


def short(tls_id):
    return tls_id if len(tls_id) <= 14 else tls_id[:10] + '...'


# -- 2 and 3. live audit and replay ------------------------------------------------------
class Recorder(object):
    """Per-second, read-only measurements, credited to junctions (see the module docstring)."""

    def __init__(self, env, traci, subset_phase, red_cap_s):
        self.env, self.traci = env, traci
        self.tls = list(env.rl_tls_list)
        self.lane_j = {}
        for j, t in enumerate(self.tls):
            for lane in env.lanes[t]:
                for phys in env.physical_lanes(t, lane):
                    self.lane_j[phys] = j
        self.queue_lanes = [[lane for lane in env.lanes[t]] for t in self.tls]   # evaluator's lanes
        self.n_lanes = sum(len(l) for l in self.queue_lanes)
        self.subset_phase = subset_phase          # j -> set of subset phases of aliased pairs
        self.actions, self.changed, self.step_second = None, [], 0
        self.yellow = int(env.yellow_s)
        # per lane (graph node order)
        self.node_of = {}
        for n, (t, lane) in enumerate(zip(env.node_tls, env.node_lane)):
            for phys in env.physical_lanes(t, lane):
                self.node_of[phys] = n
        self.node_j = [self.tls.index(t) for t in env.node_tls]
        n_nodes = len(env.node_lane)
        self.lane_halt = np.zeros(n_nodes)       # sum over seconds of whole-lane halting
        self.lane_max_wait = np.zeros(n_nodes)
        self.lane_long = np.zeros(n_nodes)       # vehicle-seconds waiting >= LONG_WAIT_S
        self.lane_max_red = np.zeros(n_nodes)    # longest time without green (ArlgosEnv's tracker)
        self.lane_red_capped = np.zeros(n_nodes)  # decisions with red_t at its cap
        self.red_cap_s = red_cap_s
        self.decisions = 0
        j_count = len(self.tls)
        self.seconds = 0
        self.halt = np.zeros(j_count)             # sum over seconds of halting on j's lanes
        self.wait = np.zeros(j_count + 1)         # sum over seconds of (sum of waits on j's lanes / vehicles)
        self.max_wait = np.zeros(j_count + 1)
        self.long_wait = np.zeros(j_count + 1)    # vehicle-seconds with a wait >= LONG_WAIT_S
        self.veh_seconds = np.zeros(j_count + 1)
        self.subset_seconds = np.zeros(j_count)
        self.subset_halt = np.zeros(j_count)      # halting on j's lanes while j serves a subset phase

    def second(self):
        tr = self.traci
        lane_h = np.array([tr.lane.getLastStepHaltingNumber(l) for l in self.env.node_lane], dtype=float)
        self.lane_halt += lane_h
        h = np.zeros(len(self.tls))
        np.add.at(h, self.node_j, lane_h)        # = the sum over each junction's queue lanes
        self.halt += h
        vehicles = tr.vehicle.getIDList()
        if vehicles:
            other = len(self.tls)
            w_sum = np.zeros(len(self.tls) + 1)
            for v in vehicles:
                lane = tr.vehicle.getLaneID(v)
                j = self.lane_j.get(lane, other)
                w = tr.vehicle.getWaitingTime(v)
                w_sum[j] += w
                self.veh_seconds[j] += 1
                if w > self.max_wait[j]:
                    self.max_wait[j] = w
                if w >= LONG_WAIT_S:
                    self.long_wait[j] += 1
                n = self.node_of.get(lane)
                if n is not None:
                    self.lane_max_wait[n] = max(self.lane_max_wait[n], w)
                    if w >= LONG_WAIT_S:
                        self.lane_long[n] += 1
            self.wait += w_sum / len(vehicles)
        if self.actions is not None:
            for j, a in enumerate(self.actions):
                in_yellow = self.changed[j] and self.step_second < self.yellow
                if int(a) in self.subset_phase.get(j, ()) and not in_yellow:
                    self.subset_seconds[j] += 1
                    self.subset_halt[j] += h[j]
            self.step_second += 1
        self.seconds += 1

    def decision(self):
        """After every step: ArlgosEnv's own red-time tracker (what the agent's red_t is made of)."""
        red = np.concatenate([self.env.red_time[t] for t in self.tls])
        self.lane_max_red = np.maximum(self.lane_max_red, red)
        self.lane_red_capped += red >= self.red_cap_s
        self.decisions += 1

    def set_actions(self, actions, prev):
        """The actions of the step about to run; a changed junction shows yellow first."""
        self.actions = actions
        self.changed = [prev is not None and a != prev[j] for j, a in enumerate(actions)]
        self.step_second = 0


def replay(args):
    run = os.path.abspath(os.path.join(REPO, args.run) if not os.path.isabs(args.run) else args.run)
    with open(os.path.join(run, 'config.json')) as f:
        cfg = json.load(f)
    net = cfg['net']
    os.environ['UNICORN_NET'] = net
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    import torch
    from parameters import SUMO_PARAMS
    from common import machine_info, use_env_sumo
    from env.arlgos_env import ArlgosEnv, DISTANCE, TIME_CAP, TIME_NORM
    from env.sumo_api import traci
    from models.Arlgos import ArlgosAgent
    use_env_sumo()
    torch.set_num_threads(1)
    seeds = list(args.seeds)
    clash = sorted(set(seeds) & set(SUMO_PARAMS.TEST_SEEDS))
    if clash:
        raise SystemExit('{} are test seeds: the diagnosis runs on validation seeds only'.format(clash))

    config = load_config(net)
    rule = cfg.get('lane_weight', 'max')                 # the run's own rule (T15, 2026-11-23)
    audit = phase_audit(config, rule)

    class DiagEnv(ArlgosEnv):
        recorder = None

        def step_a_sumo_step(self):
            super(DiagEnv, self).step_a_sumo_step()
            if self.recorder is not None:
                self.recorder.second()

    env = DiagEnv.from_run_config(cfg, args.server, test=False)
    spec = env.graph_spec()
    agent = ArlgosAgent(spec, episodes=1, seed=0, algo=cfg['algo'], conv=cfg['conv'], inter=not cfg['no_inter'],
                        agent_id=not cfg['no_agent_id'], hidden=cfg['hidden'], mixer_embed=cfg['mixer_embed'])
    ckpt = torch.load(os.path.join(run, 'model', args.checkpoint), map_location='cpu')
    agent.load_policy(ckpt)
    tls_list = list(env.rl_tls_list)
    subset_phase = {j: {pair['subset'] for pair in audit[t]['aliased'] if pair['subset'] is not None}
                    for j, t in enumerate(tls_list)}

    # -- live audit, part 2 --
    checks = []

    def check(name, ok, detail=''):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
        print('[{}] {}{}'.format('PASS' if ok else 'FAIL', name, ('  ' + detail) if detail else ''), flush=True)

    env.reset_seeded(seeds[0])
    order_bad, lanes_bad = [], []
    for t in tls_list:
        tls = env.tls_dict[t]
        links = traci.trafficlight.getControlledLinks(t)
        sumo_moves = [(l[0][0], l[0][1]) if l else None for l in links]
        if sumo_moves != list(tls.traffic_movement_list) or any(len(c) != len(links) for c in tls.action_space):
            order_bad.append(t)
        controlled = {m[0] for m in sumo_moves if m}
        nodes = {p for lane in env.lanes[t] for p in env.physical_lanes(t, lane)}
        if controlled != nodes:
            lanes_bad.append(t)
    traci.close()
    check('sumo_link_order', not order_bad, 'differs at {}'.format(order_bad) if order_bad else
          'SUMO\'s controlled links = traffic_movement_list at all {} junctions'.format(len(tls_list)))
    check('every_controlled_lane_is_a_node', not lanes_bad, str(lanes_bad) if lanes_bad else '')
    same_w = all(np.allclose(env.phase_lane_w[t], np.array(audit[t]['lane_weights'])) for t in tls_list)
    check('audit_matches_env_lane_weights', same_w)
    pad_ok = True
    for j, t in enumerate(tls_list):
        n = audit[t]['n_phases']
        pad_ok &= bool(spec['phase_mask'][j, :n].all() and not spec['phase_mask'][j, n:].any()
                       and not spec['phase_w'][j, n:].any())
    check('padded_phases_zero_and_masked', pad_ok)
    pool = agent.net.phase_pool.numpy()
    same_summary = all(np.allclose(pool[j, pair['phases'][0]], pool[j, pair['phases'][1]])
                       for j, t in enumerate(tls_list) for pair in audit[t]['aliased'])
    n_pairs = sum(len(audit[t]['aliased']) for t in tls_list)
    check('aliased_pairs_identical_in_model', same_summary,
          '{} aliased pair(s): the model\'s phase summary e_(j,p) is identical within each pair'.format(n_pairs))

    # -- replay, part 3 --
    def greedy_actions():
        detector = env.tls_map_dataset != 'resco'        # evaluator_non_rl.py's choice
        return np.array([env.tls_dict[t].get_greedy_action(detector=detector) for t in tls_list], dtype=np.int64)

    results = OrderedDict()
    light_mismatch, padded_chosen = 0, 0
    for name in ('ARLGOS-v2', 'Greedy'):
        rec = Recorder(env, traci, subset_phase, TIME_NORM * TIME_CAP)
        env.recorder = rec
        use = np.zeros((len(tls_list), env.max_phases))
        switches = np.zeros(len(tls_list))
        dp_queue, wall = [], []
        for seed in seeds:
            t0 = time.time()
            obs = env.flat_obs(env.reset_seeded(seed))
            cur, done, prev = env.current_phases(), False, None
            while not done:
                for j, t in enumerate(tls_list):          # the light SUMO shows = the tracked phase
                    if traci.trafficlight.getRedYellowGreenState(t) != env.tls_dict[t].action_space[cur[j]]:
                        light_mismatch += 1
                actions = agent.greedy(obs, cur) if name == 'ARLGOS-v2' else greedy_actions()
                for j, a in enumerate(actions):
                    if a >= audit[tls_list[j]]['n_phases']:
                        padded_chosen += 1
                    use[j, a] += 1
                    if prev is not None and a != prev[j]:
                        switches[j] += 1
                rec.set_actions(actions, prev)
                prev = actions
                nxt, _, done, _ = env.step(OrderedDict((t, int(actions[j])) for j, t in enumerate(tls_list)))
                rec.decision()
                obs, cur = env.flat_obs(nxt), env.current_phases()
            dp_queue.append(env.mean_queue())
            wall.append(round(time.time() - t0, 1))
            print('  {} seed {}: decision-point queue {:.4f} ({:.0f} s)'.format(name, seed, dp_queue[-1], wall[-1]),
                  flush=True)
        env.recorder = None
        s = float(rec.seconds)
        results[name] = {
            'decision_point_queue': dp_queue, 'decision_point_queue_mean': float(np.mean(dp_queue)),
            'per_second_queue': float(rec.halt.sum() / rec.n_lanes / s),
            'queue_contribution': (rec.halt / rec.n_lanes / s).tolist(),
            'per_second_wait': float(rec.wait.sum() / s),
            'wait_contribution': (rec.wait / s).tolist(),
            'max_wait': rec.max_wait.tolist(),
            'long_wait_share': (rec.long_wait / np.maximum(rec.veh_seconds, 1)).tolist(),
            'phase_use': (use / use.sum(axis=1, keepdims=True)).tolist(),
            'switches_per_hour': (switches / len(seeds)).tolist(),
            'subset_share': (rec.subset_seconds / s).tolist(),
            'halting_per_lane_in_subset': [float(rec.subset_halt[j] / rec.subset_seconds[j] / len(rec.queue_lanes[j]))
                                           if rec.subset_seconds[j] else None for j in range(len(tls_list))],
            'halting_per_lane': [float(rec.halt[j] / s / len(rec.queue_lanes[j])) for j in range(len(tls_list))],
            'lane_halting': (rec.lane_halt / s).tolist(),
            'lane_max_wait': rec.lane_max_wait.tolist(),
            'lane_long_wait_per_hour': (rec.lane_long / len(seeds)).tolist(),
            'lane_max_red': rec.lane_max_red.tolist(),
            'lane_red_at_cap_share': (rec.lane_red_capped / max(rec.decisions, 1)).tolist(),
            'episode_seconds': wall}
    check('light_shown_equals_tracked_phase', light_mismatch == 0,
          '{} mismatches over all decisions of both controllers'.format(light_mismatch))
    check('no_padded_phase_chosen', padded_chosen == 0)
    logged = ckpt.get('val_queue')
    if logged is not None and list(cfg.get('val_seeds', [])) == seeds:
        got = results['ARLGOS-v2']['decision_point_queue_mean']
        check('replay_reproduces_logged_validation', abs(got - logged) < 1e-6,
              'replay {:.6f} vs logged {:.6f} (checkpoint episode {})'.format(got, logged, ckpt.get('episode')))

    out = os.path.join(REPO, 'results', 'T15')
    os.makedirs(out, exist_ok=True)
    run_name = os.path.basename(run)
    # the first diagnosed run keeps its 2026-11-23 file name; any other run gets its own file
    stem = os.path.join(out, 'diag_{}'.format(net if run_name == 'arlgos_{}_seed1'.format(net) else run_name))
    record = {'run': run_name, 'net': net, 'checkpoint': args.checkpoint, 'lane_weight': rule,
              'checkpoint_episode': ckpt.get('episode'), 'logged_val_queue': logged, 'seeds': seeds,
              'tls': tls_list, 'node_tls': list(env.node_tls), 'node_lane': list(env.node_lane),
              'lane_length': [env.lane_length(t, l) for t, l in zip(env.node_tls, env.node_lane)],
              'camera_m': DISTANCE, 'red_cap_s': TIME_NORM * TIME_CAP,
              'audit': audit, 'all_networks': audit_all_networks(), 'checks': checks,
              'results': results, 'machine': machine_info(),
              'sumo_client': 'libsumo' if os.environ.get('UNICORN_LIBSUMO') == '1' else 'traci'}
    with open(stem + '.json', 'w') as f:
        json.dump(record, f, indent=1, default=float)
    md = report(record)
    with open(stem + '.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)
    print('wrote', stem + '.md')


def report(r):
    tls, audit, res = r['tls'], r['audit'], r['results']
    a, g = res['ARLGOS-v2'], res['Greedy']
    md = ['# Per-junction diagnosis: {} on `{}`'.format(r['run'], r['net']), '',
          'Checkpoint `{}` (episode {}), greedy, vs Unicorn\'s Greedy controller, on validation seeds {} '
          '(the test seeds are not used). Lane weights: `{}`. Written by `arlgos/diag_junctions.py`.'.format(
              r['checkpoint'], r['checkpoint_episode'], ', '.join(str(s) for s in r['seeds']),
              r.get('lane_weight', 'max')), '',
          '## Checks', '']
    md += ['* [{}] {}{}'.format('PASS' if c['ok'] else 'FAIL', c['name'], (': ' + c['detail']) if c['detail'] else '')
           for c in r['checks']]
    md += ['', '## Whole network', '',
           '| | ARLGOS-v2 | Greedy | ARLGOS-v2 - Greedy |', '|---|---|---|---|',
           '| Queue at decision points (the validation score) | {:.4f} | {:.4f} | {:+.4f} |'.format(
               a['decision_point_queue_mean'], g['decision_point_queue_mean'],
               a['decision_point_queue_mean'] - g['decision_point_queue_mean']),
           '| Queue every second (the test metric) | {:.4f} | {:.4f} | {:+.4f} |'.format(
               a['per_second_queue'], g['per_second_queue'], a['per_second_queue'] - g['per_second_queue']),
           '| Waiting time every second ("intersection delay", s) | {:.3f} | {:.3f} | {:+.3f} |'.format(
               a['per_second_wait'], g['per_second_wait'], a['per_second_wait'] - g['per_second_wait']), '',
           '## Per junction', '',
           'Queue and waiting-time columns are each junction\'s share of the network value, so they add up to '
           'the row above. Aliased = phase pairs with identical lane weights (subset phase in brackets).', '',
           '| Junction | Phases | Lanes | Aliased | Queue share A / G | A - G | Wait share A / G (s) | '
           'Max wait A / G (s) | Switches per h A / G |', '|---|---|---|---|---|---|---|---|---|']
    gap = a['per_second_queue'] - g['per_second_queue']
    for j, t in enumerate(tls):
        al = ', '.join('{}={} [{}]'.format(p['phases'][0], p['phases'][1], p['subset']) for p in audit[t]['aliased']) or '-'
        dq = a['queue_contribution'][j] - g['queue_contribution'][j]
        md.append('| {} | {} | {} | {} | {:.4f} / {:.4f} | {:+.4f} | {:.3f} / {:.3f} | {:.0f} / {:.0f} | {:.0f} / {:.0f} |'.format(
            short(t), audit[t]['n_phases'], audit[t]['n_lanes'], al, a['queue_contribution'][j],
            g['queue_contribution'][j], dq, a['wait_contribution'][j],
            g['wait_contribution'][j], a['max_wait'][j], g['max_wait'][j], a['switches_per_hour'][j],
            g['switches_per_hour'][j]))
    k = len(tls)
    md.append('| elsewhere | | | | | | {:.3f} / {:.3f} | {:.0f} / {:.0f} | |'.format(
        a['wait_contribution'][k], g['wait_contribution'][k], a['max_wait'][k], g['max_wait'][k]))
    md += ['', '"A - G" = the junction\'s part of ARLGOS-v2 minus Greedy on the per-second queue: negative = '
           'ARLGOS-v2 has fewer halted cars there. The column adds up to {:+.4f}.'.format(gap), '',
           '## Per lane', '',
           '"Seen" = the share of the lane the 50 m camera covers. "red_t at cap" = the share of decisions at which '
           'the lane had been red for at least {:.0f} s, where the agent\'s red_t input stops growing. Waiting = '
           'vehicle-seconds per hour with a wait of {:.0f} s or more.'.format(r['red_cap_s'], LONG_WAIT_S), '',
           '| Junction | Lane | Length (m) | Seen | Halting A / G | Longest red A / G (s) | red_t at cap A / G | '
           'Max wait A / G (s) | Waiting >= {:.0f} s A / G |'.format(LONG_WAIT_S),
           '|---|---|---|---|---|---|---|---|---|']
    for n, (t, lane) in enumerate(zip(r['node_tls'], r['node_lane'])):
        length = r['lane_length'][n]
        md.append('| {} | `{}` | {:.0f} | {:.0f} % | {:.2f} / {:.2f} | {:.0f} / {:.0f} | {:.0f} % / {:.0f} % | '
                  '{:.0f} / {:.0f} | {:.0f} / {:.0f} |'.format(
                      short(t), lane, length, 100 * min(length, r['camera_m']) / length,
                      a['lane_halting'][n], g['lane_halting'][n], a['lane_max_red'][n], g['lane_max_red'][n],
                      100 * a['lane_red_at_cap_share'][n], 100 * g['lane_red_at_cap_share'][n],
                      a['lane_max_wait'][n], g['lane_max_wait'][n],
                      a['lane_long_wait_per_hour'][n], g['lane_long_wait_per_hour'][n]))
    md += ['', '## Phase use (share of decisions)', '',
           '| Junction | ARLGOS-v2 | Greedy | Time in the subset phase A / G | Halting per lane during it A / G |',
           '|---|---|---|---|---|']
    for j, t in enumerate(tls):
        n = audit[t]['n_phases']
        fmt = lambda x: ' '.join('{:.0f}%'.format(100 * v) for v in x[j][:n])
        sub = ('{:.0f} % / {:.0f} %'.format(100 * a['subset_share'][j], 100 * g['subset_share'][j])
               if audit[t]['aliased'] else '-')
        hs = lambda x: '{:.2f}'.format(x) if x is not None else '-'
        md.append('| {} | {} | {} | {} | {} |'.format(
            short(t), fmt(a['phase_use']), fmt(g['phase_use']), sub,
            '{} / {}'.format(hs(a['halting_per_lane_in_subset'][j]), hs(g['halting_per_lane_in_subset'][j]))
            if audit[t]['aliased'] else '-'))
    md += ['', '## Phase strings of the junctions with aliased phases', '']
    for t in tls:
        if audit[t]['aliased']:
            md.append('* `{}`: '.format(t) + ', '.join('{} `{}` lanes {}'.format(
                p, code, [round(x, 1) for x in audit[t]['lane_weights'][p]]) for p, code in enumerate(audit[t]['phases'])))
    md += ['', '## The same audit on every network config in the fork (static)', '',
           '| Network | Junctions | Phases | Aliased pairs (max) | Junctions with one (max) | Aliased pairs (mean) | '
           'action_space differs from valid_phase_list |',
           '|---|---|---|---|---|---|---|']
    md += ['| {net} | {junctions} | {phases} | {aliased_pairs} | {aliased_junctions} | {aliased_pairs_mean} | '
           '{action_space_differs} |'.format(**row) for row in r['all_networks']]
    return '\n'.join(md) + '\n'


# -- self-test (static, no simulation) -----------------------------------------------------
def selftest():
    shared = {'incoming_lane_list': ['a_0', 'b_0'],
              'lane_links': {'a_0': ['x_0', 'y_0', 'z_0', 'w_0'], 'b_0': ['x_1', 'y_1', 'z_1', 'w_1']},
              'valid_phase_list': ['GGggGGgg', 'rrGGrrGG'], 'action_space': ['GGggGGgg', 'rrGGrrGG']}
    distinct = {'incoming_lane_list': ['a_0', 'b_0'], 'lane_links': {'a_0': ['x_0'], 'b_0': ['y_0']},
                'valid_phase_list': ['Gr', 'rG'], 'action_space': ['Gr', 'rG']}
    syn = phase_audit(OrderedDict([('shared', shared), ('distinct', distinct)]))
    cologne = phase_audit(load_config('cologne_network_8'))
    fenglin = phase_audit(load_config('shaoxing_network_7'))
    c_pairs = [p for a in cologne.values() for p in a['aliased']]
    syn_mean = phase_audit(OrderedDict([('shared', shared)]), 'mean')
    mean_pairs = {net: sum(len(a['aliased']) for a in phase_audit(load_config(net), 'mean').values())
                  for net in ('cologne_network_8', 'monaco_network_30')}
    checks = [
        ('shared_lane_pair_aliased', syn['shared']['aliased'] == [{'phases': [0, 1], 'subset': 1}]),
        ('distinct_phases_not_aliased', syn['distinct']['aliased'] == []),
        ('cologne8_six_pairs_at_five_junctions', len(c_pairs) == 6 and sum(1 for a in cologne.values() if a['aliased']) == 5
         and all(p['subset'] is not None for p in c_pairs)),
        ('fenglin_no_aliased_pairs', not any(a['aliased'] for a in fenglin.values())),
        ('cologne8_action_space_is_valid_phase_list', all(a['action_space_is_valid_phase_list'] for a in cologne.values())),
        # lane_weight 'mean' (2026-11-23): GGgg -> 0.75, rrGG -> 0.5, and no aliasing left
        ('mean_rule_values', np.allclose(syn_mean['shared']['lane_weights'], [[0.75, 0.75], [0.5, 0.5]])
         and syn_mean['shared']['aliased'] == []),
        ('mean_rule_no_aliasing_cologne8_monaco', mean_pairs == {'cologne_network_8': 0, 'monaco_network_30': 0}),
    ]
    for name, ok in checks:
        print('[{}] diag_{}'.format('PASS' if ok else 'FAIL', name))
    return all(ok for _, ok in checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='Train_MATSC/<name>')
    ap.add_argument('--checkpoint', default='best_val.pt')
    ap.add_argument('--seeds', type=int, nargs='*', default=list(VAL_SEEDS))
    ap.add_argument('--server', type=int, default=450)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.run:
        ap.error('give --run or --selftest')
    replay(args)


if __name__ == '__main__':
    main()
