"""ARLGOS (T9): build maps/corridor_3 from the ARLGOS repo's scenario (docs/T9_design.md 2.1).

    python arlgos/build_corridor_map.py [--arlgos ../ARLGOS]

1. Copies the scenario files (ARLGOS sumo/: corridor.sumocfg, .net.xml, .rou.xml, .det.xml
   and corridor_map.json) into maps/corridor_3 (Decision 1: copied, so the fork stays
   self-contained on the lab PC), and checks that the copy has ARLGOS's scenario
   fingerprint (sumo_env.scenario_fingerprint, the value in ARLGOS's config.json files).
2. Writes maps/corridor_3/corridor_3_config.json in Unicorn's per-junction format (as
   maps/net_config_saver.py does for Cologne8), read from a running SUMO:
     * traffic movements in SUMO's link-index order, so movement i is character i of a
       light string (Tls.get_traffic_movements_phase rebuilds them from lane_links);
     * action_space = valid_phase_list = [EW green, NS green], built with ARLGOS's own
       state_string() from corridor_map.json, so action 0 = EW green, 1 = NS green;
     * neighbours along the corridor, phase type '2.0' (2 phases), lane lengths / speeds.
3. Writes maps/corridor_3/arlgos_corridor.json: the fingerprint, the camera detectors
   (per camera and per lane) and the light string of every signal state, which
   env/corridor.py reads. The env refuses to run if the files' fingerprint changes.
"""
import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import REPO, use_env_sumo                                   # noqa: E402

NEIGHBOURS = {'C3': ['C2'], 'C2': ['C3', 'C1'], 'C1': ['C2']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arlgos', default=os.path.join(os.path.dirname(REPO), 'ARLGOS'),
                    help='the ARLGOS repo (default: next to this fork)')
    args = ap.parse_args()
    os.chdir(REPO)
    use_env_sumo()
    import traci
    from env.corridor import (CORRIDOR_DIR, CFG, MAP, ARLGOS_JSON, NET_NAME, scenario_files,
                              scenario_fingerprint, sumo_binary)
    from env.corridor_signals import STATES, state_string, FIXED_DELTA_SECONDS

    src_cfg = os.path.join(args.arlgos, 'sumo', 'corridor.sumocfg')
    src_map = os.path.join(args.arlgos, 'sumo', 'corridor_map.json')
    os.makedirs(CORRIDOR_DIR, exist_ok=True)
    for p in scenario_files(src_cfg) + [src_map]:
        shutil.copyfile(p, os.path.join(CORRIDOR_DIR, os.path.basename(p)))
    fp_src, fp = scenario_fingerprint(src_cfg, src_map), scenario_fingerprint(CFG, MAP)
    if fp != fp_src:
        raise SystemExit('copy fingerprint {} != source {}'.format(fp, fp_src))
    print('copied {} files, fingerprint {}'.format(len(scenario_files(CFG)) + 1, fp))

    with open(MAP) as f:
        cmap = json.load(f)
    camera_order = list(cmap['cameras'])
    tls_order = []
    for lbl in camera_order:
        tid = lbl.split('_')[0]
        if tid not in tls_order:
            tls_order.append(tid)
    lane_detector = OrderedDict()
    for el in ET.parse(os.path.join(CORRIDOR_DIR, 'corridor.det.xml')).getroot().iter('laneAreaDetector'):
        lane_detector[el.get('lane')] = el.get('id')
    for lbl, dets in cmap['cameras'].items():
        for d in dets:
            if d not in lane_detector.values():
                raise SystemExit('camera {} detector {} not in corridor.det.xml'.format(lbl, d))

    traci.start([sumo_binary('sumo'), '-c', CFG, '--step-length', repr(FIXED_DELTA_SECONDS),
                 '--seed', '0', '--no-warnings', 'true'])
    try:
        config = OrderedDict()
        for tid in tls_order:
            links = traci.trafficlight.getControlledLinks(tid)
            movements = []
            for i, entry in enumerate(links):
                pairs = {(a, b) for a, b, _via in entry}
                if len(pairs) != 1:
                    raise SystemExit('{} link {}: expected one in/out lane pair, got {}'.format(tid, i, pairs))
                movements.append(pairs.pop())
            lane_links = OrderedDict()
            for a, b in movements:
                lane_links.setdefault(a, []).append(b)
            if [(a, b) for a, bs in lane_links.items() for b in bs] != movements:
                raise SystemExit('{}: the links of one in-lane are not contiguous; Unicorn\'s Tls '
                                 'would number its movements differently from SUMO'.format(tid))
            incoming = list(lane_links)
            outgoing = []
            for _, b in movements:
                if b not in outgoing:
                    outgoing.append(b)
            tls_map = cmap['tls'][tid]
            ew_green = state_string(tls_map, 'RED', 'GREEN')
            ns_green = state_string(tls_map, 'GREEN', 'RED')
            if not len(ew_green) == len(ns_green) == len(movements):
                raise SystemExit('{}: light strings have {} characters for {} links'.format(
                    tid, len(ew_green), len(movements)))
            lengths, speeds = OrderedDict(), OrderedDict()
            for lane in incoming + outgoing:
                lengths[lane] = traci.lane.getLength(lane)
                speeds[lane] = traci.lane.getMaxSpeed(lane)
            config[tid] = OrderedDict([
                ('incoming_lane_list', incoming),
                ('outgoing_lane_list', outgoing),
                ('incoming_detector_list', list(incoming)),      # lane ids, as net_config_saver
                ('outgoing_detector_list', list(outgoing)),
                ('lane_links', lane_links),
                ('num_links_lane', [float(traci.lane.getLinkNumber(l)) for l in incoming]),
                ('action_space', [ew_green, ns_green]),
                ('neighbor_list', NEIGHBOURS[tid]),
                ('phase_lane_state', None),
                ('phase_type', '2.0'),
                ('phase_type_vec', [0, 0, 0, 0]),
                ('tls_position_vec', list(traci.junction.getPosition(tid))),
                ('lane_length_dict', lengths),
                ('lane_max_speed_dict', speeds),
                ('valid_phase_list', [ew_green, ns_green]),
            ])
            print('{}: {} movements, {} in-lanes, {} out-lanes, EW {} / NS {}'.format(
                tid, len(movements), len(incoming), len(outgoing), ew_green, ns_green))
        version = traci.getVersion()[1]
    finally:
        traci.close()

    with open(os.path.join(CORRIDOR_DIR, '{}_config.json'.format(NET_NAME)), 'w') as f:
        json.dump(config, f, indent=1)
    info = OrderedDict([
        ('fingerprint', fp), ('source', os.path.abspath(args.arlgos)), ('sumo', version),
        ('tls_order', tls_order), ('camera_order', camera_order), ('cameras', cmap['cameras']),
        ('lane_detector', lane_detector),
        ('state_strings', {tid: {'|'.join(k): state_string(cmap['tls'][tid], *k) for k in STATES}
                           for tid in tls_order}),
    ])
    with open(ARLGOS_JSON, 'w') as f:
        json.dump(info, f, indent=1)
    print('wrote {}_config.json and {}'.format(NET_NAME, os.path.basename(ARLGOS_JSON)))


if __name__ == '__main__':
    main()
