"""ARLGOS (T9, Decision 7): which SUMO client Unicorn's environment code talks to.

Default (as published): traci, one socket round trip per call.
UNICORN_LIBSUMO=1: libsumo, the same API with SUMO running inside this process (about 2.5x
faster on the corridor). It needs Python >= 3.9 (environments/unicorn_libsumo.yml). Each Ray
worker is its own process, so each still gets its own simulation.

Speed only: arlgos/corridor_parity.py shows identical demand, light strings, counts and
metrics either way.
"""
import os

if os.environ.get('UNICORN_LIBSUMO') == '1':
    import libsumo as traci                 # noqa: F401
else:
    import traci                            # noqa: F401
