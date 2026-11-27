"""ARLGOS (T9): the corridor's signal rules, ported from the ARLGOS repo so that the fork
stays self-contained (docs/T9_design.md, Decision 1).

Sources, copied without behavioural change:
  * timing constants   ARLGOS shared.py         (FIXED_DELTA_SECONDS ... DECISION_STEPS)
  * SignalController   ARLGOS signal_control.py (the CARLA parts left out)
  * STATES, state_string  ARLGOS sumo_env.py

arlgos/corridor_parity.py checks that a corridor episode driven through this port shows
the same light strings, tick by tick, as one driven by ARLGOS's own SumoEnv.

Phase order per junction (never varies):
    EW_GREEN -> EW_YELLOW -> ALL_RED -> NS_GREEN -> NS_YELLOW -> ALL_RED -> ...
A green ends on the first tick where time_in_phase >= its target (30..90 s);
yellow (10 s) and all-red (5 s) always run in full.
"""

FIXED_DELTA_SECONDS = 0.05          # SUMO step length
YELLOW_DURATION = 10
ALL_RED_DURATION = 5
STATIC_GREEN = 60                   # fixed-time green, and the warm-up's green
STATIC_CYCLE = 2 * (STATIC_GREEN + YELLOW_DURATION + ALL_RED_DURATION)   # 150 s
RL_GREEN_OPTIONS = [30, 40, 50, 60, 70, 80, 90]
DECISION_STEPS = 200                # 10 s between decisions
MIN_GREEN = float(min(RL_GREEN_OPTIONS))
MAX_GREEN = float(max(RL_GREEN_OPTIONS))

EW_GREEN, EW_YELLOW, NS_GREEN, NS_YELLOW, ALL_RED = 'EW_GREEN', 'EW_YELLOW', 'NS_GREEN', 'NS_YELLOW', 'ALL_RED'
_SEQUENCE = (EW_GREEN, EW_YELLOW, ALL_RED, NS_GREEN, NS_YELLOW, ALL_RED)
GREEN_PHASES = (EW_GREEN, NS_GREEN)
YELLOW_PHASES = (EW_YELLOW, NS_YELLOW)
_EPS = 1e-9

# Every (ns_state, ew_state) a SignalController can be in, named by its phase.
STATES = {('RED', 'GREEN'): 'EW_GREEN', ('RED', 'YELLOW'): 'EW_YELLOW',
          ('ALLRED', 'ALLRED'): 'ALL_RED',
          ('GREEN', 'RED'): 'NS_GREEN', ('YELLOW', 'RED'): 'NS_YELLOW'}


def state_string(tls, ns, ew):
    """The SUMO light string for one junction in state (ns, ew); tls is one entry of
    corridor_map.json["tls"] (per link: its signal group and its green character)."""
    out = []
    for grp, ch in zip(tls['groups'], tls['green']):
        st = ns if grp == 'NS' else ew if grp == 'EW' else 'RED'
        out.append(ch if st == 'GREEN' else 'y' if st == 'YELLOW' else 'r')
    return ''.join(out)


class SignalController:
    """One junction's signal state machine (ARLGOS signal_control.SignalController)."""

    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self, green_target=STATIC_GREEN):
        self._i = 0
        self._t = 0.0
        self._target = float(green_target)
        self._next_target = float(green_target)

    @property
    def phase(self):
        return _SEQUENCE[self._i]

    @property
    def time_in_phase(self):
        return self._t

    @property
    def green_target(self):
        return self._target if self.phase in GREEN_PHASES else self._next_target

    def state_for(self, approach):
        ph = self.phase
        if ph == ALL_RED:
            return 'ALLRED'
        if approach == 'EW':
            return 'GREEN' if ph == EW_GREEN else 'YELLOW' if ph == EW_YELLOW else 'RED'
        if approach == 'NS':
            return 'GREEN' if ph == NS_GREEN else 'YELLOW' if ph == NS_YELLOW else 'RED'
        raise ValueError('approach must be NS or EW, got {!r}'.format(approach))

    def is_green(self, approach):
        return self.state_for(approach) == 'GREEN'

    @property
    def ew_state(self):
        return self.state_for('EW')

    @property
    def ns_state(self):
        return self.state_for('NS')

    def phase_duration(self):
        ph = self.phase
        if ph in GREEN_PHASES:
            return self._target
        if ph in YELLOW_PHASES:
            return float(YELLOW_DURATION)
        return float(ALL_RED_DURATION)

    def set_target(self, g):
        g = float(g)
        self._next_target = g
        if self.phase in GREEN_PHASES:
            self._target = g

    def step(self, dt):
        self._t += dt
        if self._t + _EPS >= self.phase_duration():
            self._advance()

    def _advance(self):
        self._i = (self._i + 1) % len(_SEQUENCE)
        self._t = 0.0
        if self.phase in GREEN_PHASES:
            self._target = self._next_target

    def next_green(self):
        """The green this junction shows next: the current one while green, else the one
        the fixed order brings after this yellow / all-red."""
        if self.phase in GREEN_PHASES:
            return self.phase
        return NS_GREEN if self._i in (1, 2) else EW_GREEN

    def __repr__(self):                                  # pragma: no cover
        return '<SignalController {} {} t={:.2f}/{:.0f}s>'.format(
            self.name, self.phase, self._t, self.phase_duration())


class CorridorSignals:
    """The three controllers, in the order C3, C2, C1."""

    def __init__(self, names=('C3', 'C2', 'C1')):
        self.controllers = [SignalController(n) for n in names]

    def __len__(self):
        return len(self.controllers)

    def __iter__(self):
        return iter(self.controllers)

    def __getitem__(self, i):
        return self.controllers[i]

    def reset(self):
        for c in self.controllers:
            c.reset()

    def set_targets(self, greens):
        for c, g in zip(self.controllers, greens):
            c.set_target(g)

    def step(self, dt):
        for c in self.controllers:
            c.step(dt)
