import os
import random
import xml.etree.ElementTree as ET

# ARLGOS (T15F): two opt-in switches, read from the environment so that Unicorn's Ray workers,
# ARLGOS-v2 and the evaluators all see the same setting without new arguments.
#
#   UNICORN_FCD=1          write the full trajectory file fcd.xml, as upstream does. Default: off.
#                          It is output only (nothing reads it), about 270 MB per one-hour episode,
#                          and every worker writes the SAME fcd.xml next to the map. Switching it off
#                          changes no vehicle and no metric (arlgos/fenglin_check.py, check "fcd").
#   UNICORN_DEMAND=heldout a new traffic day per simulation seed, built from the fixed route file:
#                          every vehicle keeps its route, its departure time moves by a uniform
#                          random amount in [-J, +J] s (UNICORN_DEMAND_JITTER, default 60), and it
#                          is kept with probability UNICORN_DEMAND_KEEP (default 1.0). Default:
#                          'published', the fixed route file, exactly as upstream.
BASE_ROUTES = 'fenglin_y2z_t.rou.xml'
EPISODE_END = 3600.0
_BASE = {}                                   # path -> [(vehicle id, depart, route edges)]


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def fcd_wanted():
    return os.environ.get('UNICORN_FCD', '0') == '1'


def demand_settings():
    """('published' | 'heldout', jitter seconds, keep probability)."""
    mode = os.environ.get('UNICORN_DEMAND', 'published') or 'published'
    if mode not in ('published', 'heldout'):
        raise ValueError('UNICORN_DEMAND must be published or heldout, got {!r}'.format(mode))
    jitter = float(os.environ.get('UNICORN_DEMAND_JITTER', '60'))
    keep = float(os.environ.get('UNICORN_DEMAND_KEEP', '1.0'))
    if jitter < 0 or not 0.0 < keep <= 1.0:
        raise ValueError('need UNICORN_DEMAND_JITTER >= 0 and 0 < UNICORN_DEMAND_KEEP <= 1')
    return mode, jitter, keep


def base_vehicles(path):
    """The fixed route file's vehicles, in file order (read once per process)."""
    if path not in _BASE:
        root = ET.parse(os.path.join(path, BASE_ROUTES)).getroot()
        _BASE[path] = [(v.get('id'), float(v.get('depart')), v.find('route').get('edges'))
                       for v in root.iter('vehicle')]
    return _BASE[path]


def heldout_vehicles(path, seed, jitter, keep):
    """One held-out traffic day: the same vehicles and routes, departure times jittered, and
    optionally thinned. Depends on (seed, jitter, keep) alone: random.Random(int) gives the same
    stream on every Python version and machine. Sorted by departure, as SUMO requires."""
    rng = random.Random(int(seed))
    out = []
    for k, (vid, depart, edges) in enumerate(base_vehicles(path)):
        shift = rng.uniform(-jitter, jitter)  # always drawn, so `keep` does not change the shifts
        kept = rng.random() < keep
        if kept:
            t = min(max(depart + shift, 0.0), EPISODE_END - 1.0)
            out.append((round(t, 2), k, vid, edges))
    out.sort()
    return out


def write_heldout_routes(path, seed, thread, jitter, keep):
    name = 'heldout_{}.rou.xml'.format('x' if thread is None else int(thread))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<!-- ARLGOS T15F held-out demand: seed {}, jitter {} s, keep {} -->'.format(seed, jitter, keep),
             '<routes>']
    for t, _, vid, edges in heldout_vehicles(path, seed, jitter, keep):
        lines.append('    <vehicle id="{}" depart="{:.2f}">\n        <route edges="{}"/>\n    </vehicle>'.format(
            vid, t, edges))
    lines.append('</routes>\n')
    write_file(os.path.join(path, name), '\n'.join(lines))
    return name


def gen_shaoxing_sumocfg_file(path, seed, thread):
    if thread is None:
        sumocfg_file = 'shaoxing_network_7.sumocfg'
    else:
        sumocfg_file = '/shaoxing_network_7_{}.sumocfg'.format(int(thread))

    mode, jitter, keep = demand_settings()
    if mode == 'heldout':
        if seed is None:
            raise ValueError('held-out demand needs the simulation seed')
        routes = write_heldout_routes(path, seed, thread, jitter, keep)
    else:
        routes = BASE_ROUTES

    str_content = '<?xml version="1.0" encoding="UTF-8"?>\n\n'
    str_content += '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n'
    str_content += '    <input>\n'
    str_content += '        <net-file value="fenglin_y2z_t.net.xml"/>\n'
    str_content += '        <route-files value="{}"/>\n'.format(routes)
    str_content += '        <additional-files value="E1_add.xml, e2detectors.add.xml, E3_add.xml"/>\n\n'
    str_content += '    </input>\n\n'
    str_content += '    <output>\n'
    str_content += '        <write-license value="true"/>\n'
    if fcd_wanted():
        str_content += '        <fcd-output value="fcd.xml"/>\n'
        str_content += '        <fcd-output.signals value="true"/>\n'
    str_content += '    </output>\n\n'
    str_content += '    <processing>\n'
    str_content += '        <default.speeddev value="0"/>\n'
    str_content += '    </processing>\n\n'
    str_content += '    <report>\n'
    str_content += '        <duration-log.disable value="true"/>\n'
    str_content += '        <no-step-log value="true"/>\n'
    str_content += '    </report>\n\n'
    str_content += '</configuration>'

    write_file(path+sumocfg_file, str_content)

    print('Generate Shaoxing sumocfg file: {}.'.format(path+sumocfg_file))


if __name__ == '__main__':
    gen_shaoxing_sumocfg_file(path='/', seed=0, thread=0)
