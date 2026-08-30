#!/usr/bin/env python3
"""Genera config/mission_<mondo>.yaml da mujoco_world.WORLDS.

I goal suggeriti vivono nel registro dei mondi: scriverli a mano nei file
missione li fa divergere in silenzio appena si ritocca una geometria (e' gia'
successo con dead_end, allungato da 8 a 12 m, e con horseshoe da 5 a 12 m).
Qui si rigenerano tutti:

    python3 src/g1_sim/scripts/gen_missions.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from g1_sim.mujoco_world import WORLDS  # noqa: E402

OUT = os.path.join(_ROOT, "config")

TEMPLATE = """# GENERATO DA scripts/gen_missions.py — non modificare a mano.
# Mondo: {name} — {desc}
#
#   ros2 launch g1_sim g1_a_star_mpc.launch.py world:={name} \\
#        use_mission:=true \\
#        mission_file:=$(ros2 pkg prefix g1_sim)/share/g1_sim/config/{fname}
#
# reach_radius > goal_reached_radius (0.25) del profilo G1: sotto quella soglia
# a_star_node smette di ripianificare, quindi una soglia piu' stretta bloccherebbe
# la missione a un passo dal traguardo.
frame_id: odom
reach_radius:     0.35
leg_timeout_sec: {timeout:.1f}

waypoints:
  - {{ x: {gx:6.2f}, y: {gy:6.2f}, yaw: 0.0, name: "goal" }}
"""


def main() -> int:
    for name, info in sorted(WORLDS.items()):
        gx, gy = info["goal"]
        sx, sy, _ = info["spawn"]
        diretta = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
        # margine largo: una trappola concava puo' costare 3x la distanza
        # diretta, a ~0.28 m/s medi. Meglio abbondare che vedere fallire una
        # tratta per timeout e credere che sia il pianificatore.
        # I mondi con muro lungo richiedono di costeggiare per decine di metri
        # (e nel caso peggiore di rifarli all'indietro), quindi il tempo non si
        # ricava dalla distanza diretta: si dichiara nel registro dei mondi.
        timeout = float(info.get("timeout",
                                 max(180.0, round(3.5 * diretta / 0.28, -1))))
        fname = f"mission_{name}.yaml"
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(TEMPLATE.format(name=name, desc=info["desc"], fname=fname,
                                     gx=gx, gy=gy, timeout=timeout))
        print(f"  {fname:34s} goal=({gx:6.2f},{gy:6.2f})  timeout={timeout:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
