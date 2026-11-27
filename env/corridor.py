"""ARLGOS (T9): the ARLGOS corridor (3 junctions C3, C2, C1) as a Unicorn environment.

docs/T9.md in the ARLGOS repo explains every choice; docs/T9_design.md is the approved design.

What is ARLGOS's, so that both methods see the same traffic:
  * the scenario files, copied to maps/corridor_3 by arlgos/build_corridor_map.py and
    fingerprinted (the env refuses a copy whose fingerprint changed);
  * SUMO's arguments (ARLGOS sumo_env.SumoEnv._args): 0.05 s steps, --seed = demand seed,
    --scale 1.0 (the route file is frozen at 0.875), and the same output files (T7);
  * the episode: 300 s of fixed-time warm-up, then 90 decisions 10 s apart (900 s);
  * the signal rules: ARLGOS's SignalController (env/corridor_signals.py): cyclic order,
    green 30-90 s, 10 s yellow + 5 s all-red;
  * the score reported back: AQL = mean camera count at the 90 decision points.

What stays Unicorn's: the network, PPO, the phase vectors, the 8 features per movement
(Eq. 1) in sensing 'full', the reward (get_truncated_queue_reward_resco_network, in- and
out-lane halting within 50 m), gamma 0.95 per decision.

Unicorn's action = a phase index per junction, 0 = EW green, 1 = NS green
(build_corridor_map.py writes action_space in that order). It drives the
SignalController the way pressure.max_pressure_greens does in ARLGOS (Decision 2):
  * in a green, the phase showing -> keep (target 90 s, revisited at the next decision);
    the other phase -> switch (target 30 s: end now if the green has run 30 s, else then);
  * in yellow / all-red the next green is fixed by the cycle, so the phase mask leaves
    only that phase: the action is forced (probability 1, log-probability 0).

Sensing (Decision 4), per traffic movement [signal, stopped_in, stopped_out, moving_in,
moving_out, occupancy_in, occupancy_out, out_lane_signalised]:
  * 'full'  : Unicorn's own observe_resco_network (perfect sensors, 50 m);
  * 'camera': what ARLGOS's cameras give. stopped_in := the camera count on the in-lane
    (ARLGOS's 50 m lane-area detectors, the same ones SumoEnv reads), moving_in := 0,
    occupancy := 0; out-lanes only where another junction's camera watches them (the
    internal links), else 0. The signal and out-lane flag stay: a controller knows its
    own lights. Counts use Unicorn's normalisation (/20).
"""
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
from env.sumo_api import traci   # traci, or libsumo with UNICORN_LIBSUMO=1

from env.corridor_signals import (FIXED_DELTA_SECONDS, STATIC_CYCLE, DECISION_STEPS, MIN_GREEN,
                                  MAX_GREEN, STATES, GREEN_PHASES, EW_GREEN, NS_GREEN, CorridorSignals)
from env.matsc import MATSC_Env, draw_train_seed
from parameters import SUMO_PARAMS

NET_NAME = 'corridor_3'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRIDOR_DIR = os.path.join(REPO, 'maps', NET_NAME)
CFG = os.path.join(CORRIDOR_DIR, 'corridor.sumocfg')
MAP = os.path.join(CORRIDOR_DIR, 'corridor_map.json')
ARLGOS_JSON = os.path.join(CORRIDOR_DIR, 'arlgos_corridor.json')

WARMUP_S = 2 * STATIC_CYCLE                    # 300 s, whole fixed-time cycles
CONTROL_S = 900
N_DECISIONS = int(round(CONTROL_S / (DECISION_STEPS * FIXED_DELTA_SECONDS)))   # 90, 10 s apart
PHASE_OF_ACTION = (EW_GREEN, NS_GREEN)         # action index -> green it asks for
SENSING_MODES = ('full', 'camera')


# -- scenario identity: ARLGOS sumo_env.scenario_files / scenario_fingerprint, unchanged --
def scenario_files(cfg=CFG):
    root = ET.parse(cfg).getroot()
    base = os.path.dirname(os.path.abspath(cfg))
    files = [os.path.abspath(cfg)]
    for tag in ('net-file', 'route-files', 'additional-files'):
        for el in root.iter(tag):
            for name in el.get('value', '').split(','):
                if name.strip():
                    files.append(os.path.join(base, name.strip()))
    return files


def scenario_fingerprint(cfg=CFG, map_path=MAP):
    h = hashlib.sha1()
    for p in scenario_files(cfg) + [os.path.abspath(map_path)]:
        h.update(os.path.basename(p).encode())
        with open(p, 'rb') as f:
            data = f.read().replace(b'\r\n', b'\n')
        if p.lower().endswith(('.xml', '.sumocfg')):
            data = re.sub(rb'<!--.*?-->', b'', data, flags=re.S)
        h.update(data)
    return h.hexdigest()[:12]


def sumo_binary(name='sumo'):
    """The sumo executable of this env's eclipse-sumo (as ARLGOS's sumo_env.sumo_binary)."""
    try:
        import sumo
        exe = os.path.join(sumo.SUMO_HOME, 'bin', name + ('.exe' if os.name == 'nt' else ''))
        if os.path.exists(exe):
            return exe
    except ImportError:
        pass
    from sumolib import checkBinary
    return checkBinary(name)


def detectors_1s(path, out):
    """ARLGOS SumoEnv._detectors_1s: a copy of a detector file whose lane-area detectors
    write 1 s intervals to out/det.xml (ids, lanes, positions, lengths untouched)."""
    tree = ET.parse(path)
    dets = list(tree.getroot().iter('laneAreaDetector'))
    if not dets:
        return path
    for el in dets:
        el.set('period', '1')
        el.set('file', os.path.join(out, 'det.xml'))
    copy = os.path.join(out, os.path.basename(path))
    tree.write(copy)
    return copy


class CorridorEnv(MATSC_Env):
    def __init__(self, server_number, test=False, sensing=None, outputs=None):
        self.sensing = sensing or getattr(SUMO_PARAMS, 'SENSING', 'full')
        if self.sensing not in SENSING_MODES:
            raise ValueError('sensing must be one of {}, got {!r}'.format(SENSING_MODES, self.sensing))
        self.outputs = os.path.abspath(outputs) if outputs else None
        self._started = False
        self._cache = {}
        self.aql_series = []
        super(CorridorEnv, self).__init__(server_number, test)

    # -- set-up ------------------------------------------------------------------------
    def initialization(self):
        self.net_file_name = NET_NAME
        self.net_file_path = CORRIDOR_DIR
        self.net_config_path = os.path.join(CORRIDOR_DIR, '{}_config.json'.format(NET_NAME))
        super(CorridorEnv, self).initialization()       # Tls objects, padding dims, tls_map_dataset

        with open(ARLGOS_JSON) as f:
            info = json.load(f)
        now = scenario_fingerprint()
        if now != info['fingerprint']:
            raise RuntimeError('maps/{} fingerprint {} != {} recorded by build_corridor_map.py: '
                               'the scenario files changed; rebuild the map'.format(NET_NAME, now,
                                                                                    info['fingerprint']))
        self.fingerprint = now
        if self.tls_list != info['tls_order']:
            raise RuntimeError('config order {} != {}'.format(self.tls_list, info['tls_order']))
        self.state_strings = {tls: {tuple(k.split('|')): v for k, v in info['state_strings'][tls].items()}
                              for tls in self.tls_list}
        self.lane_detector = info['lane_detector']                  # lane -> camera detector
        self.cam_dets = [info['cameras'][lbl] for lbl in info['camera_order']]
        for tls in self.tls_list:
            missing = [l for l in self.tls_dict[tls].incoming_lane_list if l not in self.lane_detector]
            if missing:
                raise RuntimeError('{}: in-lanes without a camera detector: {}'.format(tls, missing))
        self.signals = CorridorSignals(self.tls_list)

    # -- SUMO --------------------------------------------------------------------------
    def sumo_args(self, seed):
        """ARLGOS SumoEnv._args(seed, tripinfo=None) with demand_scale 1.0, quiet, no GUI."""
        a = ['-c', CFG, '--step-length', repr(FIXED_DELTA_SECONDS), '--seed', str(int(seed)),
             '--scale', repr(1.0), '--no-warnings', 'true']
        if self.outputs:
            out = os.path.join(self.outputs, str(int(seed)))
            os.makedirs(out, exist_ok=True)
            a += ['--summary-output', os.path.join(out, 'summary.xml'),
                  '--summary-output.period', '1',
                  '--fcd-output', os.path.join(out, 'fcd.xml'),
                  '--fcd-output.attributes', 'speed,lane,pos,waiting',
                  '--device.fcd.period', '1', '--device.fcd.begin', str(WARMUP_S),
                  '--precision', '4',
                  '--tripinfo-output', os.path.join(out, 'tripinfo.xml'),
                  '--tripinfo-output.write-unfinished', 'true']
            root = ET.parse(CFG).getroot()
            adds = [os.path.join(CORRIDOR_DIR, n.strip()) for el in root.iter('additional-files')
                    for n in el.get('value', '').split(',') if n.strip()]
            a += ['--additional-files', ','.join(detectors_1s(p, out) for p in adds)]
        return a

    def set_world(self, seed, write_unfinished=False):
        args = self.sumo_args(seed)
        if self._started:
            traci.load(args)
        else:
            traci.start([sumo_binary('sumo')] + args)      # a free port: workers never collide
            self._started = True
        return 0

    def close(self):
        if self._started:
            self._started = False
            try:
                traci.close()
            except Exception:                                # pragma: no cover
                pass

    def tick(self):
        """One 0.05 s step, in ARLGOS's order: write changed lights -> simulationStep ->
        advance the SignalControllers (SumoEnv.tick)."""
        for ctrl, tid in zip(self.signals, self.tls_list):
            key = (ctrl.ns_state, ctrl.ew_state)
            if self._cache.get(tid) != key:
                traci.trafficlight.setRedYellowGreenState(tid, self.state_strings[tid][key])
                self._cache[tid] = key
        traci.simulationStep()
        self.signals.step(FIXED_DELTA_SECONDS)
        self.sumo_step += 1

    # -- episodes ----------------------------------------------------------------------
    def start_episode(self, seed):
        """Load SUMO with this demand seed, run the 300 s fixed-time warm-up (60 s greens),
        and leave every junction at EW green, t = 0 (ARLGOS SumoEnv.reset)."""
        self.set_world(seed)
        self.seed = int(seed)
        self._cache = {}
        self.signals.reset()
        for _ in range(int(round(WARMUP_S / FIXED_DELTA_SECONDS))):
            self.tick()
        for c in self.signals:
            if not (c.phase == EW_GREEN and c.time_in_phase < FIXED_DELTA_SECONDS):
                raise RuntimeError('warm-up ended at {} {} t={:.3f}, expected EW_GREEN t=0'.format(
                    c.name, c.phase, c.time_in_phase))
        self.signals.reset()
        self.reset_vars()                       # rl_step 0, start_simulation_step = 300 s
        self.aql_series = []
        self.record_counts()
        return self.observe()

    def reset(self):
        if self.random_seed:
            self.seed = draw_train_seed()       # never a reserved validation / test seed
        else:
            self.seed += 1
        return self.start_episode(self.seed)

    def reset_test(self, curr_episode, seed):
        assert self.test is True
        self.curr_episode = curr_episode
        return self.start_episode(seed)

    def camera_counts(self):
        """Vehicles on each camera's detectors (ARLGOS SumoEnv.detect, count_mode 'vehicles')."""
        la = traci.lanearea
        return [int(sum(la.getLastStepVehicleNumber(d) for d in dets)) for dets in self.cam_dets]

    def record_counts(self):
        self.aql_series.append(float(np.mean(self.camera_counts())))

    def aql(self):
        """ARLGOS's AQL: mean camera count over the decision points (t = 0, 10, ..., 890 s)."""
        return float(np.mean(self.aql_series[:N_DECISIONS]))

    def targets_for(self, action_dict):
        """Green targets (s) for CorridorSignals.set_targets, in tls order (Decision 2)."""
        greens = []
        for ctrl, tls in zip(self.signals, self.tls_list):
            if ctrl.phase in GREEN_PHASES:
                want = PHASE_OF_ACTION[int(action_dict[tls])]
                greens.append(MAX_GREEN if want == ctrl.phase else MIN_GREEN)
            else:
                greens.append(MAX_GREEN)        # arms the next green; revisited 10 s later
        return greens

    def forced_actions(self):
        """{tls: action} for the junctions in yellow / all-red, whose next green is fixed."""
        return {tls: PHASE_OF_ACTION.index(ctrl.next_green())
                for ctrl, tls in zip(self.signals, self.tls_list) if ctrl.phase not in GREEN_PHASES}

    def step(self, action_dict):
        switches = sum(1 for ctrl, tls in zip(self.signals, self.tls_list)
                       if ctrl.phase in GREEN_PHASES and PHASE_OF_ACTION[int(action_dict[tls])] != ctrl.phase)
        self.signals.set_targets(self.targets_for(action_dict))
        for _ in range(DECISION_STEPS):
            self.tick()
        self.rl_step += 1
        self.pre_actions_dict = action_dict
        done = self.rl_step >= N_DECISIONS
        if not done:
            self.record_counts()
        next_obs_dict = self.observe()
        rewards_dict = self.calculate_reward()
        n = len(self.rl_tls_list)
        return next_obs_dict, rewards_dict, done, [sum(rewards_dict.values()) / n, switches / n]

    def check_terminal(self):
        return self.rl_step >= N_DECISIONS

    # -- what the agent sees -----------------------------------------------------------
    def get_phase_vec_mask_dict(self):
        phase_vec_dict, phase_mask_dict = super(CorridorEnv, self).get_phase_vec_mask_dict()
        for tls, action in self.forced_actions().items():
            mask = np.ones_like(phase_mask_dict[tls])
            mask[action] = 0
            phase_mask_dict[tls] = mask
        return phase_vec_dict, phase_mask_dict

    def observe(self):
        out = {}
        for tls in self.rl_tls_list:
            obs = self.observe_camera(tls) if self.sensing == 'camera' else self.observe_full(tls)
            pad = self.tls_max_movement_dim - obs.shape[0]
            out[tls] = np.concatenate((obs, np.zeros((pad, obs.shape[1]))), 0)
        return out

    @staticmethod
    def signal_of(ch):
        return 1 if ch == 'G' else 0.5 if ch == 'g' else 0

    def observe_full(self, tls_id):
        """Tls.observe_resco_network, with one change: the signal of each movement is read
        from the light string SUMO shows. Upstream maps that string through action_space,
        which only holds greens; here a decision can fall in yellow or all-red (every
        movement 0 then, as a red is). During a green the two are identical
        (arlgos/check_fork.py checks it)."""
        tls = self.tls_dict[tls_id]
        distance = 50
        observation = []
        real = traci.trafficlight.getRedYellowGreenState(tls_id)
        for i, (in_lane, out_lane) in enumerate(tls.traffic_movement_list):
            phase_signal = self.signal_of(real[i])
            stopped_in, moving_in, occ_in = 0, 0, 0
            length = tls.lane_length_dict[in_lane]
            for v in traci.lane.getLastStepVehicleIDs(in_lane):
                if (length - traci.vehicle.getLanePosition(v)) <= distance:
                    if traci.vehicle.getSpeed(v) < 0.1:
                        stopped_in += 1
                    else:
                        moving_in += 1
                    occ_in += traci.vehicle.getLength(v)
            stopped_out, moving_out, occ_out = 0, 0, 0
            length = tls.lane_length_dict[out_lane]
            for v in traci.lane.getLastStepVehicleIDs(out_lane):
                if (length - traci.vehicle.getLanePosition(v)) <= distance:
                    if traci.vehicle.getSpeed(v) < 0.1:
                        stopped_out += 1
                    else:
                        moving_out += 1
                    occ_out += traci.vehicle.getLength(v)
            observation.append([phase_signal, stopped_in / 20, stopped_out / 20, moving_in / 20,
                                moving_out / 20, occ_in / distance, occ_out / distance,
                                self.outgoing_lane_state_dict[out_lane]])
        return np.array(observation)

    def observe_camera(self, tls_id):
        tls = self.tls_dict[tls_id]
        real = traci.trafficlight.getRedYellowGreenState(tls_id)
        la = traci.lanearea
        counts = {}

        def count(lane):
            if lane not in counts:
                det = self.lane_detector.get(lane)
                counts[lane] = la.getLastStepVehicleNumber(det) if det else 0
            return counts[lane]

        observation = []
        for i, (in_lane, out_lane) in enumerate(tls.traffic_movement_list):
            observation.append([self.signal_of(real[i]), count(in_lane) / 20, count(out_lane) / 20,
                                0, 0, 0, 0, self.outgoing_lane_state_dict[out_lane]])
        return np.array(observation)

    def calculate_reward(self):
        """Unicorn's own reward for lane-based networks, unchanged, in both sensing modes."""
        return {tls: self.tls_dict[tls].get_truncated_queue_reward_resco_network(
                    regional_reward=self.regional_reward) for tls in self.rl_tls_list}
