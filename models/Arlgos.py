"""ARLGOS-v2 for arbitrary signalised networks (ROADMAP T14).

The A1 recipe (ARLGOS repo: arlgos_model.py, marl.py, rl_training.py), with the two pieces that
were hard-wired to the 3-junction corridor made general:

  graph      A1: 12 cameras, a fixed 12 x 12 adjacency.
             here: one node per incoming lane, built from the network (env/arlgos_env.py
             graph_spec), with three kinds of neighbour instead of one:
               nb    the other lanes of the same junction         (A1's intersection block)
               up    lanes whose traffic arrives here next        (the platoon that is coming)
               down  lanes this lane's traffic goes to next       (the storage it needs)

  layer      A1 (--gnn_conv self_nb):  h_i' = relu( W_self h_i + W_nb mean_{j in N(i)} h_j )
             here ('flow', default):   h_i' = relu( W_self h_i + W_nb mean_nb h_j
                                                    + W_up mean_up h_j + W_down mean_down h_j )
             The own-features matrix W_self is what made A1 work (arm A0 without it collapsed
             to fixed-time). 'flow' keeps it and gives each neighbour kind its own matrix, so
             "a queue upstream of me" and "a queue beside me" are different inputs. conv
             'self_nb' is A1's layer exactly, on the union of the three neighbour sets.

  head       A1: 7 Q-values per junction (green durations), junctions told apart by a one-hot.
             here: one Q-value per (junction, phase), from ONE network shared by all junctions
             and all phases:
               e_p = weighted mean of the embeddings of the lanes phase p serves
               c   = mean of the embeddings of all the junction's lanes
               Q_j(p) = MLP([e_p, c, is p showing now?, one-hot(j)])
             A junction with 2 phases and one with 8 use the same weights; padded phases are
             masked out of every argmax and every random draw. This is Unicorn's idea of
             describing a phase by what it serves (its G_p vector, paper Eq. 2), used for
             Q-values instead of policy logits.

  learning   unchanged from A1: QMIX (Rashid et al. 2018) with double-Q targets,
               Q_tot = f_s(Q_1(a_1), ..., Q_J(a_J)),  dQ_tot/dQ_j >= 0,
             f's weights produced from the global state s (every node's features) by
             hypernetworks, trained on the team reward with a Huber loss:
               y = r + gamma * f_s'^-( Q_j^-(s', argmax_a Q_j(s', a)) )
             Monotonic f means the team's best joint action is every junction's own argmax.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONVS = ('flow', 'self_nb')
ALGOS = ('qmix', 'vdn', 'iql')
HIDDEN_DIM = 32            # A1's GNN widths
OUT_DIM = 32
N_LAYERS = 2
HEAD_HIDDEN = 128          # A1's Q-network width
MIXER_EMBED = 32
A1_TEAM_EXPLORE = 1.0 - 0.95 ** 3      # chance that any of A1's 3 agents explores at eps 0.05


def row_normalise(a):
    """Rows that sum to 1 (a mean over the neighbours); all-zero rows stay zero."""
    d = a.sum(axis=1, keepdims=True)
    return a / np.where(d > 0, d, 1.0)


class ArlgosNet(nn.Module):
    """Graph encoder + phase-generic Q head. forward(x, cur) -> Q of shape (B, J, Pmax).

    x   : (B, N, f_dyn) node features from ArlgosEnv.observe(), in node order
    cur : (B, J) long, the phase each junction is showing
    """

    def __init__(self, spec, conv='flow', inter=True, agent_id=True, hidden=HIDDEN_DIM, out_dim=OUT_DIM,
                 n_layers=N_LAYERS, head_hidden=HEAD_HIDDEN):
        super(ArlgosNet, self).__init__()
        if conv not in CONVS:
            raise ValueError('conv must be one of {}, got {!r}'.format(CONVS, conv))
        self.conv, self.inter, self.agent_id = conv, bool(inter), bool(agent_id)
        self.n_nodes = int(spec['a_nb'].shape[0])
        self.n_agents, self.max_phases = [int(v) for v in spec['phase_mask'].shape]
        zero = np.zeros_like(spec['a_up'])
        up, down = (spec['a_up'], spec['a_down']) if self.inter else (zero, zero)
        if conv == 'flow':
            rels = [row_normalise(spec['a_nb']), row_normalise(up), row_normalise(down)]
        else:
            rels = [row_normalise(np.clip(spec['a_nb'] + up + down, 0.0, 1.0))]
        # buffers, not parameters, and not saved: the graph is rebuilt from the map
        self.register_buffer('rels', torch.tensor(np.stack(rels), dtype=torch.float32), persistent=False)
        self.register_buffer('phase_pool', torch.tensor(row_normalise_last(spec['phase_w']), dtype=torch.float32),
                             persistent=False)
        self.register_buffer('phase_mask', torch.tensor(spec['phase_mask'], dtype=torch.bool), persistent=False)
        pool = np.zeros((self.n_agents, self.n_nodes), dtype=np.float32)
        pool[spec['node_agent'], np.arange(self.n_nodes)] = 1.0
        self.register_buffer('junction_pool', torch.tensor(row_normalise(pool), dtype=torch.float32),
                             persistent=False)
        self.register_buffer('static', torch.tensor(spec['static'], dtype=torch.float32), persistent=False)

        in_dim = int(spec['f_dyn']) + int(spec['static'].shape[1])
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        # bias-free, as every A1 run was
        self.self_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)])
        self.rel_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for _ in range(len(rels))])
            for i in range(n_layers)])
        head_in = 2 * out_dim + 1 + (self.n_agents if self.agent_id else 0)
        self.head = nn.Sequential(nn.Linear(head_in, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 1))

    def node_embeddings(self, x):
        """(B, N, out_dim): one embedding per incoming lane."""
        h = torch.cat([x, self.static.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=-1)
        for self_layer, rel_layers in zip(self.self_layers, self.rel_layers):
            z = self_layer(h)
            for r, layer in enumerate(rel_layers):
                z = z + torch.matmul(self.rels[r], layer(h))      # mean over that neighbour kind
            h = F.relu(z)
        return h

    def forward(self, x, cur):
        h = self.node_embeddings(x)
        b = h.shape[0]
        e = torch.einsum('jpn,bnd->bjpd', self.phase_pool, h)      # what each phase serves
        c = torch.einsum('jn,bnd->bjd', self.junction_pool, h)     # the whole junction
        showing = F.one_hot(cur, self.max_phases).to(h.dtype).unsqueeze(-1)
        parts = [e, c.unsqueeze(2).expand(-1, -1, self.max_phases, -1), showing]
        if self.agent_id:
            ids = torch.eye(self.n_agents, dtype=h.dtype, device=h.device)
            parts.append(ids.view(1, self.n_agents, 1, self.n_agents).expand(b, -1, self.max_phases, -1))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)

    def masked(self, q):
        """q with the padded phases at -inf, so no argmax can pick one."""
        return q.masked_fill(~self.phase_mask.unsqueeze(0), float('-inf'))


def row_normalise_last(w):
    """(J, P, N) phase weights -> each phase's weights sum to 1 (zero rows stay zero)."""
    d = w.sum(axis=-1, keepdims=True)
    return w / np.where(d > 0, d, 1.0)


class QMixer(nn.Module):
    """QMIX's monotonic mixing network, as in ARLGOS marl.py.

    qs : (B, J) chosen-action values;  s : (B, state_dim) global state.
    abs() on the hypernetworks' outputs makes every mixing weight non-negative, which is what
    gives dQ_tot/dQ_j >= 0.
    """

    def __init__(self, n_agents, state_dim, embed=MIXER_EMBED, hyper_hidden=64):
        super(QMixer, self).__init__()
        self.n_agents, self.embed = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, n_agents * embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, embed))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, qs, s):
        b = qs.size(0)
        w1 = torch.abs(self.hyper_w1(s)).view(b, self.n_agents, self.embed)
        b1 = self.hyper_b1(s).view(b, 1, self.embed)
        hidden = F.elu(torch.bmm(qs.view(b, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(s)).view(b, self.embed, 1)
        b2 = self.hyper_b2(s).view(b, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(b)


class VDNMixer(nn.Module):
    def forward(self, qs, s):
        return qs.sum(dim=-1)


class Replay(object):
    """Transition replay in fixed numpy arrays (A1: a deque of 20000 transitions)."""
    FIELDS = ('obs', 'cur', 'act', 'rew', 'obs2', 'cur2', 'done')

    def __init__(self, capacity, n_nodes, f_dyn, n_agents):
        self.capacity, self.size, self.ptr = int(capacity), 0, 0
        self.obs = np.zeros((capacity, n_nodes, f_dyn), dtype=np.float32)
        self.obs2 = np.zeros((capacity, n_nodes, f_dyn), dtype=np.float32)
        self.cur = np.zeros((capacity, n_agents), dtype=np.int64)
        self.cur2 = np.zeros((capacity, n_agents), dtype=np.int64)
        self.act = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rew = np.zeros((capacity, n_agents), dtype=np.float32)     # one per junction
        self.done = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, cur, act, rew, obs2, cur2, done):
        i = self.ptr
        self.obs[i], self.cur[i], self.act[i], self.rew[i] = obs, cur, act, rew
        self.obs2[i], self.cur2[i], self.done[i] = obs2, cur2, float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, idx):
        return tuple(torch.from_numpy(getattr(self, f)[idx]) for f in self.FIELDS)

    def state(self):
        d = {f: getattr(self, f)[:self.size].copy() for f in self.FIELDS}
        d.update(size=self.size, ptr=self.ptr, capacity=self.capacity)
        return d

    def load(self, d):
        if d['capacity'] != self.capacity:
            raise ValueError('replay capacity {} != {}'.format(d['capacity'], self.capacity))
        self.size, self.ptr = int(d['size']), int(d['ptr'])
        for f in self.FIELDS:
            getattr(self, f)[:self.size] = d[f]


class ArlgosAgent(object):
    """The learner: A1's RLAgent (rl_training.py) for the phase-generic network.

    Defaults are A1's own values (rl_training.py's argparse defaults plus run_A1.ps1's flags);
    nothing here was tuned on a benchmark network. The one rule that had to change is the
    exploration floor, see eps_floor().
    """

    def __init__(self, spec, episodes, seed=0, algo='qmix', conv='flow', inter=True, agent_id=True,
                 lr=5e-4, gamma=0.99, batch_size=32, memory_size=20000, target_update=50,
                 eps_decay=0.82, eps_min=None, pretrain_eps=2, hidden=HEAD_HIDDEN,
                 mixer_embed=MIXER_EMBED, grad_clip=1.0, bootstrap_timeout=True):
        if algo not in ALGOS:
            raise ValueError('algo must be one of {}, got {!r}'.format(ALGOS, algo))
        torch.manual_seed(seed)
        self.algo, self.gamma, self.batch_size = algo, float(gamma), int(batch_size)
        self.target_update, self.grad_clip = int(target_update), float(grad_clip)
        self.eps_decay, self.pretrain_eps = float(eps_decay), int(pretrain_eps)
        self.bootstrap_timeout = bool(bootstrap_timeout)
        self.net = ArlgosNet(spec, conv=conv, inter=inter, agent_id=agent_id, head_hidden=hidden)
        self.tgt = copy.deepcopy(self.net)
        self.n_agents, self.max_phases, self.n_nodes = self.net.n_agents, self.net.max_phases, self.net.n_nodes
        self.f_dyn = int(spec['f_dyn'])
        self.eps_min = float(eps_min) if eps_min is not None else self.eps_floor(self.n_agents)
        self.n_phases = spec['phase_mask'].sum(axis=1).astype(np.int64)        # real phases per junction
        params = list(self.net.parameters())
        self.mixer = self.tgt_mixer = None
        if algo == 'qmix':
            self.mixer = QMixer(self.n_agents, self.n_nodes * self.f_dyn, embed=mixer_embed)
        elif algo == 'vdn':
            self.mixer = VDNMixer()
        if self.mixer is not None:
            self.tgt_mixer = copy.deepcopy(self.mixer)
            params += list(self.mixer.parameters())
        for p in list(self.tgt.parameters()) + (list(self.tgt_mixer.parameters()) if self.tgt_mixer else []):
            p.requires_grad_(False)
        self.params = params
        self.opt = torch.optim.Adam(params, lr=lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=int(episodes), eta_min=1e-5)
        self.memory = Replay(memory_size, self.n_nodes, self.f_dyn, self.n_agents)
        self.rng = np.random.RandomState(seed)        # exploration and replay sampling only
        self.epsilon, self.steps = 1.0, 0
        self.best_val = float('inf')

    @staticmethod
    def eps_floor(n_agents):
        """Per-junction exploration floor that keeps the TEAM's exploration at A1's level.

        A1: 3 junctions, each random with probability 0.05, so some junction explores in
        1 - 0.95^3 = 14.3 % of decisions. With 16 junctions at 0.05 each that would be 56 %:
        the team reward would be judged under a random partner most of the time. Solving
        1 - (1 - eps)^J = 14.3 % gives eps = 0.05 for J = 3 and 0.0096 for J = 16.
        """
        return 1.0 - (1.0 - A1_TEAM_EXPLORE) ** (1.0 / n_agents)

    # -- acting ------------------------------------------------------------------------
    def q_values(self, obs, cur):
        with torch.no_grad():
            q = self.net(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(cur).unsqueeze(0))
            return self.net.masked(q)[0]

    def greedy(self, obs, cur):
        return self.q_values(obs, cur).argmax(dim=-1).numpy().astype(np.int64)

    def act(self, obs, cur, episode):
        """Per-junction epsilon-greedy over that junction's real phases (A1: select_action).
        The first pretrain_eps episodes are fully random."""
        random_all = episode < self.pretrain_eps
        actions = np.zeros(self.n_agents, dtype=np.int64) if random_all else self.greedy(obs, cur)
        for j in range(self.n_agents):
            if random_all or self.rng.random_sample() < self.epsilon:
                actions[j] = self.rng.randint(self.n_phases[j])
        return actions

    # -- learning ----------------------------------------------------------------------
    def learn(self, episode):
        """One gradient step on a replay batch (A1: train_step). Returns the loss."""
        if episode < self.pretrain_eps or self.memory.size < self.batch_size:
            return 0.0
        idx = self.rng.randint(0, self.memory.size, size=self.batch_size)
        obs, cur, act, rew, obs2, cur2, done = self.memory.sample(idx)
        q = self.net(obs, cur).gather(2, act.unsqueeze(2)).squeeze(2)                 # (B, J)
        with torch.no_grad():
            best = self.net.masked(self.net(obs2, cur2)).argmax(dim=2, keepdim=True)  # online picks
            qn = self.tgt(obs2, cur2).gather(2, best).squeeze(2)                      # target scores
        if self.mixer is None:                  # iql: every junction on its own reward
            loss = F.smooth_l1_loss(q, rew + self.gamma * qn * (1 - done).unsqueeze(1))
        else:
            q_tot = self.mixer(q, obs.reshape(obs.shape[0], -1))
            with torch.no_grad():
                y = rew.mean(dim=1) + self.gamma * (1 - done) * self.tgt_mixer(
                    qn, obs2.reshape(obs2.shape[0], -1))
            loss = F.smooth_l1_loss(q_tot, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.sync_targets()
        return float(loss.item())

    def sync_targets(self):
        self.tgt.load_state_dict(self.net.state_dict())
        if self.tgt_mixer is not None:
            self.tgt_mixer.load_state_dict(self.mixer.state_dict())

    def end_episode(self, episode):
        if episode >= self.pretrain_eps:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            self.sched.step()

    # -- saving ------------------------------------------------------------------------
    def policy_state(self):
        """What evaluation needs."""
        return {'net': self.net.state_dict(),
                'mixer': self.mixer.state_dict() if self.mixer is not None else None}

    def load_policy(self, state):
        self.net.load_state_dict(state['net'])
        if self.mixer is not None and state.get('mixer') is not None:
            self.mixer.load_state_dict(state['mixer'])

    def full_state(self):
        """Everything needed to continue training exactly where it stopped."""
        return {'policy': self.policy_state(), 'tgt': self.tgt.state_dict(),
                'tgt_mixer': self.tgt_mixer.state_dict() if self.tgt_mixer is not None else None,
                'opt': self.opt.state_dict(), 'sched': self.sched.state_dict(),
                'epsilon': self.epsilon, 'steps': self.steps, 'best_val': self.best_val,
                'rng': self.rng.get_state(), 'memory': self.memory.state()}

    def load_full(self, state):
        self.load_policy(state['policy'])
        self.tgt.load_state_dict(state['tgt'])
        if self.tgt_mixer is not None:
            self.tgt_mixer.load_state_dict(state['tgt_mixer'])
        self.opt.load_state_dict(state['opt'])
        self.sched.load_state_dict(state['sched'])
        self.epsilon, self.steps, self.best_val = state['epsilon'], state['steps'], state['best_val']
        self.rng.set_state(state['rng'])
        self.memory.load(state['memory'])

    def n_parameters(self):
        return sum(p.numel() for p in self.params)
