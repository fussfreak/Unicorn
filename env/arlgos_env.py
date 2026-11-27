"""ARLGOS (T14): Unicorn's benchmark environment, seen through ARLGOS's cameras.

ArlgosEnv is MATSC_Env with two methods replaced, observe() and calculate_reward(). Everything
that decides what the traffic does is inherited unchanged: the network and route files, the
SUMO arguments, the seeds, the phase list of every junction, green / yellow times and step().
So ARLGOS-v2 and Unicorn drive the same junctions under the same rules (ROADMAP T14: "identical
by construction"; arlgos/check_arlgos.py replays one action sequence through both classes and
compares the light strings and vehicle counts).

What the agent sees: one graph node per incoming lane of every signalised junction, with what a
camera above the stop line plus the controller's own clock can give (no outgoing lanes, no
occupancy, no speeds):

    0  count      vehicles within 50 m of the stop line / 10      (the camera count, A1's input)
    1  signal     1 protected green, 0.5 permitted green, 0 red   (under the phase showing now)
    2  phase_t    seconds the junction has held its phase / 60, capped at 3
    3  red_t      seconds since this lane last had green / 60, capped at 3 (0 while green)

Channel 3 replaces A1's cyclic order. On the corridor the phases alternate, so "time in phase"
says everything about the cycle. Here the agent may pick any phase at any step, so the state
needs to say how long each lane has been waiting, or it is not Markov and an approach can
starve unnoticed. A controller knows its own light history, so this is still camera-only.

The reward is one number per junction (the runner averages them into the team reward QMIX
trains on):

    halting (default)  r_j = -0.25 * mean over j's incoming lanes of halting vehicles within 50 m
    count              r_j = -0.25 * mean over j's incoming lanes of all vehicles within 50 m

'halting' is the in-lane half of Unicorn's own reward; its out-lane half exists to make a
junction care about its neighbours, which a team reward does by construction. 'count' is A1's
choice (a reward the cameras alone can produce) and is kept as an ablation.

How a phase is described per lane (lane_weight; T15, added 2026-11-23, user-approved):
    A lane carries one or more movements (through, left, right, possibly to several out-lanes);
    a phase string gives each movement G (1), g (0.5) or r (0). The lane's weight w_{p,l} under
    phase p is
        'max'  (default, T14)  w = the best of its movements
        'mean'                 w = the average over its movements
    w feeds the phase summary e_{j,p} = sum_l w_{p,l} h_l / sum_l w_{p,l} (models/Arlgos.py),
    the signal channel (w under the phase showing) and red_t (time since the lane was "green"):
        'max'   green <=> w > 0                (any of its movements is green)
        'mean'  green <=> w >= min(0.5, the lane's best w over all phases)
                                               (at least half of it is served, or as much as
                                                any phase ever serves it)
    With 'max', two phases that serve different movements of the same lanes ("through + permitted
    left" vs "left only" on a shared lane) get identical weights: 6 such pairs on Cologne8 at 5 of
    its 8 junctions, 20 on Monaco, none on Fenglin, Arterial 4x4 or the grids
    (arlgos/diag_junctions.py). 'mean' tells all of them apart, and a lane whose waiting cars need
    a red turn no longer looks served. 'max' stays the default, unchanged.
"""
import glob
import os
# Load Python's XML parser BEFORE libsumo: on Windows libsumo brings its own expat library, and
# once that is loaded "import pyexpat" fails ("No parsers found" when sumolib reads the network).
import xml.parsers.expat  # noqa: F401
from collections import OrderedDict

import numpy as np

from env.sumo_api import traci
from env.matsc import MATSC_Env

DISTANCE = 50.0            # camera range, the same 50 m as ARLGOS's corridor detectors
HALT_SPEED = 0.1           # m/s, SUMO's own definition of a halting vehicle
COUNT_NORM = 10.0
TIME_NORM = 60.0
TIME_CAP = 3.0
REWARD_SCALE = 0.25
F_DYN = 4                  # dynamic features per node (the list above)
REWARDS = ('halting', 'count')
LANE_WEIGHTS = ('max', 'mean')
MEAN_GREEN = 0.5           # lane_weight 'mean': a lane counts as green when at least half served


def code_weight(ch):
    """Unicorn's phase-vector convention (Tls.get_traffic_movements_phase)."""
    return 1.0 if ch == 'G' else 0.5 if ch == 'g' else 0.0


class ArlgosEnv(MATSC_Env):
    def __init__(self, server_number, test=False, reward='halting', lane_weight='max'):
        if reward not in REWARDS:
            raise ValueError('reward must be one of {}, got {!r}'.format(REWARDS, reward))
        if lane_weight not in LANE_WEIGHTS:
            raise ValueError('lane_weight must be one of {}, got {!r}'.format(LANE_WEIGHTS, lane_weight))
        self.reward_mode = reward
        self.lane_weight = lane_weight
        self._scan_key = None
        self._tracked_step = 0
        super(ArlgosEnv, self).__init__(server_number, test)
        self._build_static()
        self._reset_trackers()

    @classmethod
    def from_run_config(cls, cfg, server_number, test=False):
        """The environment a training run used, from its config.json (evaluation, diagnosis).
        Runs from before 2026-11-23 have no lane_weight key: they used 'max'."""
        return cls(server_number, test=test, reward=cfg['reward'], lane_weight=cfg.get('lane_weight', 'max'))

    # -- static structure --------------------------------------------------------------
    def _build_static(self):
        """Node order (junction by junction, lanes in the config's order) and, per junction,
        the weight of every lane under every phase (see lane_weight in the module docstring)."""
        self.green_s = float(self.sumo_steps_green_phase)
        self.yellow_s = float(self.sumo_steps_yellow_phase)
        self.node_tls, self.node_lane = [], []
        self.lanes = OrderedDict()                   # tls -> its incoming lanes (node order)
        self.phase_lane_w = OrderedDict()            # tls -> array (n_phases, n_lanes)
        self.lane_green = OrderedDict()              # tls -> bool (n_phases, n_lanes), for red_t
        for tls_id in self.rl_tls_list:
            tls = self.tls_dict[tls_id]
            lanes = list(tls.incoming_lane_list)
            self.lanes[tls_id] = lanes
            w = np.zeros((len(tls.valid_phase_list), len(lanes)), dtype=np.float32)
            for p, code in enumerate(tls.valid_phase_list):
                if self.lane_weight == 'max':
                    # one character per traffic movement; a lane takes its best movement
                    for ch, (in_lane, _) in zip(code, tls.traffic_movement_list):
                        li = lanes.index(in_lane)
                        w[p, li] = max(w[p, li], code_weight(ch))
                else:
                    # a lane takes the average of its movements
                    total = np.zeros(len(lanes))
                    n_moves = np.zeros(len(lanes))
                    for ch, (in_lane, _) in zip(code, tls.traffic_movement_list):
                        li = lanes.index(in_lane)
                        total[li] += code_weight(ch)
                        n_moves[li] += 1
                    w[p] = np.where(n_moves > 0, total / np.maximum(n_moves, 1), 0.0)
            self.phase_lane_w[tls_id] = w
            if self.lane_weight == 'max':
                self.lane_green[tls_id] = w > 0
            else:
                best = w.max(axis=0)
                self.lane_green[tls_id] = (best > 0) & (w >= np.minimum(MEAN_GREEN, best))
            for lane in lanes:
                self.node_tls.append(tls_id)
                self.node_lane.append(lane)
        self.n_nodes = len(self.node_lane)
        self.n_agents = len(self.rl_tls_list)
        self.max_phases = int(max(w.shape[0] for w in self.phase_lane_w.values()))

    def physical_lanes(self, tls_id, lane):
        """The SUMO lanes behind one configured incoming lane (Ingolstadt replaces some)."""
        return self.tls_dict[tls_id].replaced_lane_dict.get(lane, [lane])

    def graph_spec(self, max_hops=8, max_len=1500.0):
        """Everything models/Arlgos.py needs to build the graph, as plain numpy arrays.

        node_agent (N,)       junction index of every node
        a_nb   (N, N)         1 where j is another lane of i's junction
        a_up   (N, N)         1 where traffic leaving lane j can reach lane i as its next signal
        a_down (N, N)         a_up transposed: the lanes i feeds
        phase_w (J, Pmax, N)  lane weights of each phase (0 rows for padded phases)
        phase_mask (J, Pmax)  1 for a real phase
        static (N, 1)         visible length: min(lane length, 50 m) / 50 m

        a_up / a_down come from the network file: from every lane's out-edges, walk forward
        through unsignalised nodes until an incoming edge of another signalised junction (at
        most max_hops edges and max_len metres). On Cologne8 and Fenglin the junctions are
        rarely adjacent edge-to-edge, so a direct "out-lane is an in-lane" test finds almost
        nothing (6 of 33 and 0 of 59 out-lanes).
        """
        import sumolib
        n, j_count = self.n_nodes, self.n_agents
        tls_index = {t: k for k, t in enumerate(self.rl_tls_list)}
        node_agent = np.array([tls_index[t] for t in self.node_tls], dtype=np.int64)
        a_nb = (node_agent[:, None] == node_agent[None, :]).astype(np.float32)
        np.fill_diagonal(a_nb, 0.0)

        net_files = sorted(glob.glob(os.path.join(self.net_file_path, '*.net.xml')))
        if len(net_files) != 1:
            raise RuntimeError('expected one .net.xml in {}, found {}'.format(self.net_file_path, net_files))
        net = sumolib.net.readNet(net_files[0])

        def edge_of(lane):
            return net.getLane(lane).getEdge()

        edge_nodes = {}                              # incoming edge id -> node indices on it
        for i, (t, lane) in enumerate(zip(self.node_tls, self.node_lane)):
            for phys in self.physical_lanes(t, lane):
                edge_nodes.setdefault(edge_of(phys).getID(), set()).add(i)

        def next_signal_nodes(start_edge, src_agent):
            """Nodes of other junctions first met when driving on from start_edge."""
            found, seen = set(), {start_edge.getID()}
            frontier = [(start_edge, 0.0, 0)]
            while frontier:
                edge, dist, hops = frontier.pop(0)
                hit = {i for i in edge_nodes.get(edge.getID(), ()) if node_agent[i] != src_agent}
                if hit:
                    found |= hit
                    continue                         # that junction's signal is the next one
                if hops >= max_hops or dist + edge.getLength() > max_len:
                    continue
                for nxt in edge.getOutgoing():
                    if nxt.getID() not in seen:
                        seen.add(nxt.getID())
                        frontier.append((nxt, dist + edge.getLength(), hops + 1))
            return found

        a_down = np.zeros((n, n), dtype=np.float32)
        for i, (t, lane) in enumerate(zip(self.node_tls, self.node_lane)):
            tls = self.tls_dict[t]
            for out_lane in tls.lane_links[lane]:
                for phys in self.physical_lanes(t, out_lane):
                    for d in next_signal_nodes(edge_of(phys), node_agent[i]):
                        a_down[i, d] = 1.0
        a_up = a_down.T.copy()

        phase_w = np.zeros((j_count, self.max_phases, n), dtype=np.float32)
        phase_mask = np.zeros((j_count, self.max_phases), dtype=np.float32)
        static = np.zeros((n, 1), dtype=np.float32)
        offset = 0
        for k, t in enumerate(self.rl_tls_list):
            w = self.phase_lane_w[t]
            phase_w[k, :w.shape[0], offset:offset + w.shape[1]] = w
            phase_mask[k, :w.shape[0]] = 1.0
            for li, lane in enumerate(self.lanes[t]):
                static[offset + li, 0] = min(self.lane_length(t, lane), DISTANCE) / DISTANCE
            offset += w.shape[1]
        return {'net': self.net_file_name, 'tls': list(self.rl_tls_list),
                'node_lane': list(self.node_lane), 'node_agent': node_agent,
                'a_nb': a_nb, 'a_up': a_up, 'a_down': a_down,
                'phase_w': phase_w, 'phase_mask': phase_mask, 'static': static, 'f_dyn': F_DYN}

    def lane_length(self, tls_id, lane):
        tls = self.tls_dict[tls_id]
        if lane in tls.replaced_lane_dict:
            return float(tls.replaced_lane_length_dict[lane])
        return float(tls.lane_length_dict[lane])

    # -- per-episode state -------------------------------------------------------------
    def _reset_trackers(self):
        self.cur_phase = OrderedDict((t, 0) for t in self.rl_tls_list)       # reset() shows phase 0
        self.phase_time = OrderedDict((t, 0.0) for t in self.rl_tls_list)
        self.red_time = OrderedDict((t, np.zeros(len(self.lanes[t]), dtype=np.float32))
                                    for t in self.rl_tls_list)
        self._tracked_step = 0
        self._scan_key = None
        self.queue_series = []                       # evaluator-style queue at every decision

    def reset_vars(self):
        super(ArlgosEnv, self).reset_vars()
        if hasattr(self, 'lanes'):                   # not yet during MATSC_Env.__init__
            self._reset_trackers()

    def reset_seeded(self, seed):
        """reset() with a chosen simulation seed (validation); the same four calls."""
        self.seed = int(seed)
        if self.set_world(self.seed) != 0:
            raise RuntimeError('SUMO did not start (seed {}, server {})'.format(seed, self.server_number))
        for tls in self.tls_list:
            self.tls_dict[tls].set_green_phase(0, self.sumo_steps_green_phase)
        self.reset_vars()
        return self.observe()

    def _advance_trackers(self):
        """Bring the light clocks up to date with the RL step that just ran. step() sets
        pre_actions_dict and rl_step before it calls observe(), so one call = one step."""
        if self._tracked_step == self.rl_step:
            return
        if self.rl_step != self._tracked_step + 1:
            raise RuntimeError('observe() missed a step: tracked {}, now {}'.format(
                self._tracked_step, self.rl_step))
        for t in self.rl_tls_list:
            a = int(self.pre_actions_dict[t])
            if a == self.cur_phase[t]:
                self.phase_time[t] += self.green_s
            else:
                # step() shows the old phase's yellow first, except on the very first step
                self.phase_time[t] = self.green_s - (self.yellow_s if self.rl_step > 1 else 0.0)
                self.cur_phase[t] = a
            green = self.lane_green[t][a]            # 'max': w > 0, as in T14
            self.red_time[t] = np.where(green, 0.0, self.red_time[t] + self.green_s).astype(np.float32)
        self._tracked_step = self.rl_step

    def _scan(self):
        """Count every incoming lane once per decision: all vehicles and halting vehicles
        within 50 m of the stop line, and the whole-lane halting number the evaluator's
        queue metric uses (MATSC_Env.measure_traffic_step)."""
        key = (self.rl_step, traci.simulation.getTime())
        if self._scan_key == key:
            return
        self.count, self.halting = OrderedDict(), OrderedDict()
        whole = []
        for t in self.rl_tls_list:
            tls = self.tls_dict[t]
            c = np.zeros(len(self.lanes[t]), dtype=np.float32)
            h = np.zeros(len(self.lanes[t]), dtype=np.float32)
            for li, lane in enumerate(self.lanes[t]):
                if lane in tls.replaced_lane_dict:   # short replaced lanes: all of them, as Unicorn
                    for phys in tls.replaced_lane_dict[lane]:
                        c[li] += traci.lane.getLastStepVehicleNumber(phys)
                        h[li] += traci.lane.getLastStepHaltingNumber(phys)
                else:
                    length = tls.lane_length_dict[lane]
                    for v in traci.lane.getLastStepVehicleIDs(lane):
                        if (length - traci.vehicle.getLanePosition(v)) <= DISTANCE:
                            c[li] += 1
                            if traci.vehicle.getSpeed(v) < HALT_SPEED:
                                h[li] += 1
                whole.append(traci.lane.getLastStepHaltingNumber(lane))
            self.count[t], self.halting[t] = c, h
        self.queue_series.append(float(np.mean(whole)))
        self._scan_key = key

    # -- what the agent sees and is paid ------------------------------------------------
    def observe(self):
        self._advance_trackers()
        self._scan()
        out = OrderedDict()
        for t in self.rl_tls_list:
            n = len(self.lanes[t])
            x = np.zeros((n, F_DYN), dtype=np.float32)
            x[:, 0] = self.count[t] / COUNT_NORM
            x[:, 1] = self.phase_lane_w[t][self.cur_phase[t]]
            x[:, 2] = min(self.phase_time[t] / TIME_NORM, TIME_CAP)
            x[:, 3] = np.minimum(self.red_time[t] / TIME_NORM, TIME_CAP)
            out[t] = x
        return out

    def calculate_reward(self):
        self._scan()
        src = self.halting if self.reward_mode == 'halting' else self.count
        return OrderedDict((t, -REWARD_SCALE * float(np.mean(src[t]))) for t in self.rl_tls_list)

    def current_phases(self):
        """Phase index showing at every junction, in rl_tls_list order."""
        return np.array([self.cur_phase[t] for t in self.rl_tls_list], dtype=np.int64)

    def flat_obs(self, obs):
        """{tls: (lanes, F)} -> (N, F) in node order."""
        return np.concatenate([obs[t] for t in self.rl_tls_list], axis=0)

    def mean_queue(self):
        """Mean over the decision points of the evaluator's queue (halting vehicles per
        incoming lane, whole lane). The checkpoint-selection score."""
        return float(np.mean(self.queue_series)) if self.queue_series else float('nan')
