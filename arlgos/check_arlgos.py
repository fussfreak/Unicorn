"""ARLGOS (T14): self-checks for ARLGOS-v2 on Unicorn's benchmark networks.

    python arlgos/check_arlgos.py                 # everything (about 2 min)
    python arlgos/check_arlgos.py --net arterial_network_4_4 --graph-only

Graph and model, for each of Cologne8, Arterial 4x4, Grid 4x4 and Fenglin (no simulation):
   1. every lane node belongs to one junction; each junction's phase mask has exactly as many
      real phases as its action space; phase strings are as long as the movement list
   2. every real phase serves at least one lane, and every lane is served by some phase
   3. the edges between junctions exist, never join two lanes of one junction, and
      "upstream" is "downstream" reversed
   4. the network returns one Q per (junction, phase); padded phases are never chosen
   5. moving a vehicle between two lanes of one junction that different phases serve changes
      that junction's Q-values (the blind spot of the old mean convolution, ARLGOS arm A0)
   6. a queue on a lane changes the Q-values of the junction downstream of it, and exactly
      nothing there once the inter-junction edges are removed (--no_inter)
   7. QMIX's mixer is monotonic: dQ_tot/dQ_j >= 0
   8. a gradient step changes the online network only; the target follows every target_update
   9. the exploration floor is A1's 0.05 for 3 junctions and keeps the team at 14.3 % for any J
  10. full_state -> load_full restores weights, optimiser, epsilon, the replay buffer and the
      random stream
With SUMO, on Cologne8 (short episodes):
  11. ArlgosEnv and Unicorn's MATSC_Env, given the same seed and the same actions, show the
      same lights and hold the same vehicles at every step (same traffic by construction)
  12. the count channel equals an independent recount; the signal, phase-time and red-time
      channels follow the actions taken
  13. the reward is -0.25 x the mean halting count, and the env's "average reward" is the team reward
  14. a run stopped after 2 episodes and resumed gives the same train.csv as an uninterrupted one
--lane_weight (T15, 2026-11-23):
  15. graph, per network: 'max' (default) and 'mean' weights and green rules recomputed from the
      phase strings; 'mean' leaves no aliased phase pair; a bad value is refused; a config from
      before 2026-11-23 means 'max'. SUMO: a 'mean' env has the same traffic as MATSC_Env and every
      channel (red_t under the mean rule) passes checks 12-13.
  16. a 'mean' run trains end to end, records the rule, refuses a resume without it and resumes
      with it; eval_arlgos.py and diag_junctions.py rebuild the run's own env
Exits 1 if any check fails.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NETS = ('cologne_network_8', 'arterial_network_4_4', 'grid_network_4_4', 'shaoxing_network_7')
FAILED = []


def check(name, ok, detail=''):
    print('[{}] {}{}'.format('PASS' if ok else 'FAIL', name, ': ' + str(detail) if detail != '' else ''), flush=True)
    if not ok:
        FAILED.append(name)


def graph_checks(net):
    import numpy as np
    import torch
    from env.arlgos_env import ArlgosEnv, F_DYN
    from models.Arlgos import ArlgosAgent, ArlgosNet, QMixer

    env = ArlgosEnv(server_number=190, test=False)
    spec = env.graph_spec()
    n, j, pmax = env.n_nodes, env.n_agents, env.max_phases
    tag = net.split('_network')[0]

    lens_ok = all(len(code) == len(env.tls_dict[t].traffic_movement_list)
                  for t in env.rl_tls_list for code in env.tls_dict[t].valid_phase_list)
    masks_ok = all(int(spec['phase_mask'][k].sum()) == len(env.tls_dict[t].action_space)
                   for k, t in enumerate(env.rl_tls_list))
    check('{} 1. nodes, masks, phase strings ({} junctions, {} lanes, up to {} phases)'.format(tag, j, n, pmax),
          spec['node_agent'].shape == (n,) and masks_ok and lens_ok
          and spec['phase_w'].shape == (j, pmax, n), 'masks {} lengths {}'.format(masks_ok, lens_ok))

    real = spec['phase_mask'] > 0
    serves = spec['phase_w'].sum(axis=2) > 0
    own = np.array([[spec['node_agent'][i] == k for i in range(n)] for k in range(j)])
    leak = float(np.abs(spec['phase_w'] * (~own)[:, None, :]).sum())
    served = spec['phase_w'].max(axis=(0, 1)) > 0
    check('{} 2. every real phase serves a lane of its own junction; every lane has a phase'.format(tag),
          bool((serves == real).all()) and leak == 0.0 and bool(served.all()),
          '{} lanes never green'.format(int((~served).sum())) if not served.all() else '')

    same = spec['node_agent'][:, None] == spec['node_agent'][None, :]
    n_edges = int(spec['a_down'].sum())
    linked = len({(spec['node_agent'][a], spec['node_agent'][b]) for a, b in zip(*np.nonzero(spec['a_down']))})
    check('{} 3. inter-junction edges: {} lane pairs, {} junction pairs'.format(tag, n_edges, linked),
          n_edges > 0 and float((spec['a_down'] * same).sum()) == 0.0
          and bool((spec['a_up'] == spec['a_down'].T).all()))

    agent = ArlgosAgent(spec, episodes=10, seed=0)
    rng = np.random.RandomState(1)
    obs = rng.rand(n, F_DYN).astype(np.float32)
    cur = np.zeros(j, dtype=np.int64)
    q = agent.net(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(cur).unsqueeze(0))
    agent.epsilon = 1.0
    draws = np.stack([agent.act(obs, cur, episode=5) for _ in range(300)]
                     + [agent.act(obs, cur, episode=0) for _ in range(300)])
    agent.epsilon = 0.0
    greedy = agent.act(obs, cur, episode=5)
    check('{} 4. Q is (1, {}, {}); padded phases never chosen (600 random draws + greedy)'.format(tag, j, pmax),
          tuple(q.shape) == (1, j, pmax) and bool((draws < agent.n_phases[None, :]).all())
          and bool((greedy < agent.n_phases).all()))

    # 5. the A0 blind spot: two lanes of one junction that no single phase serves alike
    pair = None
    for k in range(j):
        idx = np.nonzero(spec['node_agent'] == k)[0]
        for a in idx:
            for b in idx:
                if a < b and not np.allclose(spec['phase_w'][k][:, a], spec['phase_w'][k][:, b]):
                    pair = (k, a, b)
                    break
            if pair:
                break
        if pair:
            break
    base = np.full((n, F_DYN), 0.2, dtype=np.float32)
    moved = base.copy()
    moved[pair[1], 0] += 0.3
    moved[pair[2], 0] -= 0.1
    shifted = base.copy()
    shifted[pair[1], 0] += 0.1
    shifted[pair[2], 0] += 0.1          # the same junction total as `moved`
    with torch.no_grad():
        qa = agent.net(torch.from_numpy(moved).unsqueeze(0), torch.from_numpy(cur).unsqueeze(0))[0, pair[0]]
        qb = agent.net(torch.from_numpy(shifted).unsqueeze(0), torch.from_numpy(cur).unsqueeze(0))[0, pair[0]]
    gap = float((qa - qb).abs().max())
    check('{} 5. a queue moved between two lanes of one junction is visible'.format(tag), gap > 1e-6,
          'max |dQ| {:.2e}'.format(gap))

    # 6. the path between junctions
    src, dst = [int(v[0]) for v in np.nonzero(spec['a_down'])]
    dst_agent = int(spec['node_agent'][dst])
    bumped = base.copy()
    bumped[src, 0] += 0.5
    gaps = {}
    for inter in (True, False):
        torch.manual_seed(3)
        net_ = ArlgosNet(spec, inter=inter)
        with torch.no_grad():
            q0 = net_(torch.from_numpy(base).unsqueeze(0), torch.from_numpy(cur).unsqueeze(0))[0, dst_agent]
            q1 = net_(torch.from_numpy(bumped).unsqueeze(0), torch.from_numpy(cur).unsqueeze(0))[0, dst_agent]
        gaps[inter] = float((q1 - q0).abs().max())
    check('{} 6. an upstream queue reaches the downstream junction; not without the edges'.format(tag),
          gaps[True] > 1e-6 and gaps[False] == 0.0, 'with {:.2e}, without {:.2e}'.format(gaps[True], gaps[False]))

    torch.manual_seed(0)
    mixer = QMixer(j, n * F_DYN)
    qs = torch.randn(16, j, requires_grad=True)
    mixer(qs, torch.randn(16, n * F_DYN)).sum().backward()
    check('{} 7. QMIX mixer is monotonic'.format(tag), float(qs.grad.min()) >= 0.0,
          'min dQ_tot/dQ_j {:.3e}'.format(float(qs.grad.min())))

    agent = ArlgosAgent(spec, episodes=10, seed=0, target_update=3, pretrain_eps=0)
    for _ in range(40):
        o, o2 = rng.rand(n, F_DYN).astype(np.float32), rng.rand(n, F_DYN).astype(np.float32)
        a = np.array([rng.randint(p) for p in agent.n_phases])
        agent.memory.add(o, cur, a, -rng.rand(j).astype(np.float32), o2, a, False)
    before = [p.clone() for p in agent.net.parameters()]
    tgt_before = [p.clone() for p in agent.tgt.parameters()]
    agent.learn(episode=0)
    moved_online = any(not torch.equal(a, b) for a, b in zip(before, agent.net.parameters()))
    tgt_still = all(torch.equal(a, b) for a, b in zip(tgt_before, agent.tgt.parameters()))
    agent.learn(episode=0)
    agent.learn(episode=0)
    tgt_synced = all(torch.equal(a, b) for a, b in zip(agent.net.parameters(), agent.tgt.parameters()))
    check('{} 8. a step moves the online network only; the target syncs every target_update'.format(tag),
          moved_online and tgt_still and tgt_synced,
          'online {} target-still {} synced {}'.format(moved_online, tgt_still, tgt_synced))

    floor3, floor16 = ArlgosAgent.eps_floor(3), ArlgosAgent.eps_floor(16)
    check('{} 9. exploration floor: 0.05 for 3 junctions, team rate 14.3 % for 16'.format(tag),
          abs(floor3 - 0.05) < 1e-12 and abs((1 - (1 - floor16) ** 16) - (1 - 0.95 ** 3)) < 1e-12,
          'eps(3) {:.4f} eps(16) {:.4f} eps({}) {:.4f}'.format(floor3, floor16, j, ArlgosAgent.eps_floor(j)))

    for _ in range(5):
        agent.learn(episode=0)
    agent.epsilon = 0.37
    state = agent.full_state()
    nxt = agent.rng.random_sample()
    other = ArlgosAgent(spec, episodes=10, seed=99, target_update=3, pretrain_eps=0)
    other.load_full(state)
    same_w = all(torch.equal(a, b) for a, b in zip(agent.net.state_dict().values(), other.net.state_dict().values()))
    same_m = all(torch.equal(a, b) for a, b in zip(agent.mixer.state_dict().values(),
                                                   other.mixer.state_dict().values()))
    same_buf = other.memory.size == agent.memory.size and np.array_equal(
        other.memory.obs[:other.memory.size], agent.memory.obs[:agent.memory.size])
    same_rng = nxt == other.rng.random_sample()      # both streams now stand at the same point
    same_count = other.steps == agent.steps and other.epsilon == 0.37
    la, lb = agent.learn(episode=0), other.learn(episode=0)
    check('{} 10. full_state -> load_full: weights, mixer, buffer, epsilon, random stream, next loss'.format(tag),
          same_w and same_m and same_buf and same_count and same_rng and la == lb,
          'w {} mixer {} buffer {} loss {} vs {}'.format(same_w, same_m, same_buf, la, lb))

    # 15. lane_weight (T15, 2026-11-23): both rules recomputed here from the phase strings, independently
    #     of ArlgosEnv._build_static; 'mean' leaves no two phases of a junction identical
    env_m = ArlgosEnv(server_number=189, test=False, lane_weight='mean')
    spec_m = env_m.graph_spec()
    weights_ok, green_ok, alias_max, alias_mean = True, True, 0, 0
    for k, t in enumerate(env.rl_tls_list):
        tls, lanes = env.tls_dict[t], env.lanes[t]
        n_ph = len(tls.valid_phase_list)
        ref_max, total, count = np.zeros((n_ph, len(lanes))), np.zeros((n_ph, len(lanes))), np.zeros((n_ph, len(lanes)))
        for p, code in enumerate(tls.valid_phase_list):
            for ch, (in_lane, _) in zip(code, tls.traffic_movement_list):
                li, v = lanes.index(in_lane), {'G': 1.0, 'g': 0.5}.get(ch, 0.0)
                ref_max[p, li] = max(ref_max[p, li], v)
                total[p, li] += v
                count[p, li] += 1
        ref_mean = total / np.maximum(count, 1)
        weights_ok &= bool(np.allclose(env.phase_lane_w[t], ref_max) and np.allclose(env_m.phase_lane_w[t], ref_mean)
                           and np.allclose(spec_m['phase_w'][k, :n_ph][:, spec_m['node_agent'] == k], ref_mean))
        best = ref_mean.max(axis=0)
        green_ok &= bool((env.lane_green[t] == (ref_max > 0)).all()
                         and (env_m.lane_green[t] == ((best > 0) & (ref_mean >= np.minimum(0.5, best)))).all())
        alias_max += sum(1 for p in range(n_ph) for q in range(p + 1, n_ph) if np.allclose(ref_max[p], ref_max[q]))
        alias_mean += sum(1 for p in range(n_ph) for q in range(p + 1, n_ph) if np.allclose(ref_mean[p], ref_mean[q]))
    try:
        ArlgosEnv(server_number=188, test=False, lane_weight='median')
        rejects = False
    except ValueError:
        rejects = True
    old_cfg = ArlgosEnv.from_run_config({'reward': 'halting'}, 187).lane_weight == 'max'     # runs before 11-23
    check('{} 15. --lane_weight: max (default) and mean weights and green rules; aliased pairs {} -> {}'.format(
        tag, alias_max, alias_mean),
        weights_ok and green_ok and alias_mean == 0 and rejects and old_cfg,
        'weights {} green {} rejects {} old config {}'.format(weights_ok, green_ok, rejects, old_cfg))


def sumo_checks():
    import numpy as np
    from env.arlgos_env import ArlgosEnv, DISTANCE, COUNT_NORM, REWARD_SCALE
    from env.matsc import MATSC_Env
    from env.sumo_api import traci

    seconds, seed = 300, 4242

    def plan(env, k):
        """A fixed action sequence that changes phases at different rates per junction."""
        return {t: (k // (2 + i % 3)) % len(env.tls_dict[t].action_space)
                for i, t in enumerate(env.tls_list)}

    def replay(env, on_step=None):
        env.max_sumo_step = seconds
        env.seed = seed
        assert env.set_world(seed) == 0
        for t in env.tls_list:
            env.tls_dict[t].set_green_phase(0, env.sumo_steps_green_phase)
        env.reset_vars()
        obs = env.observe()
        trace, k, done = [], 0, False
        while not done:
            actions = plan(env, k)
            # read before step(): it closes the simulation when the episode ends
            trace.append((tuple(traci.trafficlight.getRedYellowGreenState(t) for t in env.tls_list),
                          traci.vehicle.getIDCount(), traci.simulation.getArrivedNumber()))
            if on_step:
                on_step(env, k, obs, actions)
            obs, rewards, done, info = env.step(actions)
            if on_step and not done:
                on_step(env, k + 1, obs, None, rewards=rewards, info=info, taken=actions)
            k += 1
        return trace

    ref = replay(MATSC_Env(191, False))

    def make_inspect(problems, history, is_green):
        """Checks 12-13 at every step; is_green(env, t, phase, lane) is the lane_weight rule's
        idea of green, written out independently of ArlgosEnv._advance_trackers."""
        def inspect(env, k, obs, actions, rewards=None, info=None, taken=None):
            if rewards is None:
                return
            _inspect(env, k, obs, taken, rewards, info, problems, history, is_green)
        return inspect

    def _inspect(env, k, obs, taken, rewards, info, problems, history, is_green):
        history.append(taken)
        for t in env.rl_tls_list:
            tls = env.tls_dict[t]
            recount = []
            halting = []
            for lane in env.lanes[t]:
                length = tls.lane_length_dict[lane]
                near = [v for v in traci.lane.getLastStepVehicleIDs(lane)
                        if length - traci.vehicle.getLanePosition(v) <= DISTANCE]
                recount.append(len(near))
                halting.append(sum(1 for v in near if traci.vehicle.getSpeed(v) < 0.1))
            if not np.allclose(obs[t][:, 0] * COUNT_NORM, recount):
                problems.append('count {} step {}'.format(t, k))
            a = taken[t]
            if not np.allclose(obs[t][:, 1], env.phase_lane_w[t][a]):
                problems.append('signal {} step {}'.format(t, k))
            # how long has action a been held, and how long has each lane been red?
            held = 0
            for past in reversed(history):
                if past[t] != a:
                    break
                held += 1
            changed = held < len(history)
            want_t = held * env.green_s - (env.yellow_s if changed else 0.0)
            if abs(obs[t][0, 2] - min(want_t / 60.0, 3.0)) > 1e-6:
                problems.append('phase_t {} step {}: {} vs {}'.format(t, k, obs[t][0, 2], want_t / 60.0))
            for li in range(len(env.lanes[t])):
                red = 0
                for past in reversed(history):
                    if is_green(env, t, past[t], li):
                        break
                    red += 1
                if abs(obs[t][li, 3] - min(red * env.green_s / 60.0, 3.0)) > 1e-6:
                    problems.append('red_t {} lane {} step {}'.format(t, li, k))
            if abs(rewards[t] + REWARD_SCALE * float(np.mean(halting))) > 1e-6:
                problems.append('reward {} step {}'.format(t, k))
        if abs(info[0] - float(np.mean(list(rewards.values())))) > 1e-9:
            problems.append('team reward step {}'.format(k))

    problems, history = [], []
    ours = replay(ArlgosEnv(192, False), on_step=make_inspect(
        problems, history, lambda env, t, p, li: env.phase_lane_w[t][p][li] > 0))     # 'max': any movement green
    check('cologne 11. ArlgosEnv = MATSC_Env traffic: lights and vehicles identical at {} steps'.format(len(ref)),
          ref == ours and len(ref) == seconds // 15,
          '' if ref == ours else 'first difference at step {}'.format(
              next((i for i, (a, b) in enumerate(zip(ref, ours)) if a != b), '?')))
    obs_problems = [p for p in problems if not p.startswith(('reward', 'team'))]
    check('cologne 12. count, signal, phase-time and red-time channels', not obs_problems,
          '; '.join(obs_problems[:3]))
    rew_problems = [p for p in problems if p.startswith(('reward', 'team'))]
    check('cologne 13. reward = -0.25 x mean halting; env average = team reward', not rew_problems,
          '; '.join(rew_problems[:3]))

    # 15. lane_weight 'mean' (T15, 2026-11-23): the same traffic, and every channel follows the mean rule
    def mean_green(env, t, p, li):
        w = env.phase_lane_w[t]
        best = float(w[:, li].max())
        return best > 0 and float(w[p, li]) >= min(0.5, best)

    problems_m, history_m = [], []
    env_m = ArlgosEnv(195, False, lane_weight='mean')
    ours_m = replay(env_m, on_step=make_inspect(problems_m, history_m, mean_green))
    # not vacuous: lane-steps that 'max' calls green and 'mean' does not (partly served lanes)
    partial = sum(1 for taken in history_m for t in env_m.rl_tls_list for li in range(len(env_m.lanes[t]))
                  if env_m.phase_lane_w[t][taken[t]][li] > 0 and not mean_green(env_m, t, taken[t], li))
    check('cologne 15. --lane_weight mean: same traffic as MATSC_Env; count, signal, phase-time, red-time, reward '
          '({} partly served lane-steps)'.format(partial),
          ref == ours_m and not problems_m and partial > 0,
          '; '.join(problems_m[:3]) or ('traffic differs' if ref != ours_m else ''))

    # 14. stop + resume = uninterrupted
    def run(name, extra):
        cmd = [sys.executable, os.path.join(REPO, 'driver_arlgos.py'), '--net', 'cologne_network_8',
               '--seed', '5', '--name', name, '--sim_seconds', '150', '--pretrain_eps', '1',
               '--val_seeds', '5042', '--val_every', '2', '--latest_every', '2', '--batch_size', '8'] + extra
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)

    def rows(name):
        path = os.path.join(REPO, 'Train_MATSC', name, 'train.csv')
        if not os.path.exists(path):
            return None
        with open(path, newline='') as f:
            return [{k: v for k, v in r.items() if k != 'seconds'} for r in csv.DictReader(f)]

    for name in ('_check_full', '_check_part'):
        shutil.rmtree(os.path.join(REPO, 'Train_MATSC', name), ignore_errors=True)
    a = run('_check_full', ['--episodes', '4', '--server', '193'])
    b = run('_check_part', ['--episodes', '4', '--server', '194', '--stop_after', '2'])
    part2 = run('_check_part', ['--episodes', '4', '--server', '194', '--resume']) if b.returncode == 0 else None
    full, part = rows('_check_full'), rows('_check_part')
    ok = (a.returncode == 0 and part2 is not None and part2.returncode == 0
          and full is not None and full == part and len(full) == 4)
    detail = ''
    if not ok:
        err = (a.stderr if a.returncode else b.stderr if b.returncode else part2.stderr if part2 else '')
        detail = (err.strip().splitlines() or ['rows differ: {} vs {}'.format(full, part)])[-1]
    check('cologne 14. stopped after 2 episodes + resumed = uninterrupted (train.csv identical)', ok, detail)
    for name in ('_check_full', '_check_part'):
        shutil.rmtree(os.path.join(REPO, 'Train_MATSC', name), ignore_errors=True)

    # 16. a --lane_weight mean run end to end (T15, 2026-11-23): it trains and records the rule, a resume
    #     without the flag is refused, one with it continues; evaluation and diagnosis rebuild its env
    name = '_check_mean'
    shutil.rmtree(os.path.join(REPO, 'Train_MATSC', name), ignore_errors=True)
    m1 = run(name, ['--episodes', '3', '--server', '196', '--lane_weight', 'mean', '--stop_after', '2'])
    wrong = run(name, ['--episodes', '3', '--server', '196', '--resume'])
    m2 = run(name, ['--episodes', '3', '--server', '196', '--lane_weight', 'mean', '--resume'])
    cfg_path = os.path.join(REPO, 'Train_MATSC', name, 'config.json')
    recorded = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            recorded = json.load(f).get('lane_weight')
    n_rows = len(rows(name) or [])
    src_ok = True
    for script in ('eval_arlgos.py', 'diag_junctions.py'):
        with open(os.path.join(HERE, script), encoding='utf-8') as f:
            src_ok &= 'from_run_config(cfg' in f.read()
    refused = wrong.returncode != 0 and 'lane_weight' in (wrong.stdout + wrong.stderr)
    ok = m1.returncode == 0 and refused and m2.returncode == 0 and recorded == 'mean' and n_rows == 3 and src_ok
    detail = '' if ok else 'train {} refused {} resume {} recorded {} rows {} eval/diag {}{}'.format(
        m1.returncode, refused, m2.returncode, recorded, n_rows, src_ok,
        ': ' + (m1.stderr or m2.stderr).strip().splitlines()[-1] if (m1.returncode or m2.returncode) and (m1.stderr or m2.stderr) else '')
    check('cologne 16. a --lane_weight mean run: trains, records it, refuses a resume without it, resumes; '
          'eval and diagnosis rebuild its env', ok, detail)
    shutil.rmtree(os.path.join(REPO, 'Train_MATSC', name), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default=None)
    ap.add_argument('--graph-only', action='store_true')
    ap.add_argument('--sumo-only', action='store_true')
    args = ap.parse_args()
    os.chdir(REPO)
    if args.net is None:
        # parameters.py fixes the network at import, so every network gets its own process
        code = 0
        for net in NETS:
            code |= subprocess.run([sys.executable, os.path.abspath(__file__), '--net', net, '--graph-only'],
                                   cwd=REPO).returncode
        code |= subprocess.run([sys.executable, os.path.abspath(__file__), '--net', 'cologne_network_8',
                                '--sumo-only'], cwd=REPO).returncode
        print('\nARLGOS-v2 checks: {}'.format('some FAILED' if code else 'all passed'))
        sys.exit(1 if code else 0)
    os.environ['UNICORN_NET'] = args.net
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from common import use_env_sumo
    use_env_sumo()
    import torch
    torch.set_num_threads(1)
    if not args.sumo_only:
        graph_checks(args.net)
    if not args.graph_only:
        sumo_checks()
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
