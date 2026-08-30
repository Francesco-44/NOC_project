#!/usr/bin/env python3
"""
Sorgente dati dai run VERI: legge una rosbag registrata mentre il G1 naviga nel
magazzino e la trasforma in fotogrammi utilizzabili dai due pannelli.

Perche' da bag e non in diretta
-------------------------------
Il pannello 2 deve RI-RISOLVERE l'NLP per ottenere gli iterati di IPOPT: farlo
in diretta ruberebbe CPU al solutore che si sta misurando, falsando proprio la
grandezza di interesse. In replay il costo di calcolo non disturba nulla, e lo
stesso run si puo' analizzare quante volte si vuole cambiando i parametri.

Il fotogramma
-------------
I fotogrammi sono ancorati ai messaggi di /mpc/diagnostics, cioe' UNO PER CICLO
DI CONTROLLO: per ciascuno si prende il valore piu' recente di ogni altro topic,
che e' esattamente cio' che il nodo aveva a disposizione in quell'istante.

Ricostruzione esatta di x0
--------------------------
/mpc/diagnostics porta gli elementi [7..12] con lo stato iniziale passato al
solutore. Senza quelli, posizione e yaw si dedurrebbero da /mpc/predicted_path,
ma le VELOCITA' — stimate dentro mpc_node con una media esponenziale sulle
differenze di posa — non uscirebbero mai, e il solve ricostruito sarebbe un
problema diverso da quello risolto davvero.

Uso
---
    python3 viz/bag_source.py <bag>              # riepilogo del contenuto
"""
from __future__ import annotations

import os
import sys
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOPICS = {
    "pose":  "/robot_pose",
    "scan":  "/lidar/points_filtered",
    "path":  "/a_star/path",
    "pred":  "/mpc/predicted_path",
    "setpt": "/mpc/next_setpoint",
    "diag":  "/mpc/diagnostics",
    "goal":  "/global_goal",
    "cmd":   "/cmd_vel",
}


@dataclass
class Frame:
    """Uno stato completo del problema, in un ciclo di controllo."""
    t: float                      # [s] dall'inizio della bag
    x0: np.ndarray                # (6,) stato passato al solutore
    obstacles: np.ndarray         # (M, 2) punti LiDAR in odom
    path: np.ndarray | None       # (K, 2) riferimento A*
    pred: np.ndarray | None       # (N+1, 2) traiettoria predetta pubblicata
    setpoint: np.ndarray | None   # (2,)
    goal: np.ndarray | None       # (2,)
    cost: float
    solve_ms: float
    success: bool
    iterations: int

    @property
    def pose(self) -> np.ndarray:
        return self.x0[:3]


def _quat_yaw(q) -> float:
    import math
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_bag(path: str) -> dict:
    """{chiave: [(t_ns, msg), ...]} per i topic di interesse."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {v: k for k, v in TOPICS.items() if v in types}
    if not wanted:
        raise SystemExit(f"la bag non contiene nessuno dei topic attesi:\n"
                         f"  attesi:  {sorted(TOPICS.values())}\n"
                         f"  trovati: {sorted(types)}")

    out = {k: [] for k in TOPICS}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        key = wanted.get(topic)
        if key is None:
            continue
        out[key].append((t_ns, deserialize_message(data, get_message(types[topic]))))
    return out


def _latest(series, t_ns):
    """Ultimo messaggio non successivo a t_ns."""
    if not series:
        return None
    i = bisect_right([s[0] for s in series], t_ns) - 1
    return series[i][1] if i >= 0 else None


def frames(bag: dict) -> list[Frame]:
    from sensor_msgs_py import point_cloud2 as pc2

    diag = bag["diag"]
    if not diag:
        raise SystemExit("la bag non contiene /mpc/diagnostics: senza quello non "
                         "si sa in quali istanti l'MPC ha risolto")
    t0 = diag[0][0]
    out = []
    for t_ns, d in diag:
        v = list(d.data)
        if len(v) < 14:
            raise SystemExit(
                f"/mpc/diagnostics ha {len(v)} campi, ne servono 14. La bag e' "
                "stata registrata con una versione precedente di mpc_node, che "
                "non pubblicava lo stato iniziale del solutore: va rifatta.")
        x0 = np.array(v[7:13], dtype=float)

        cloud = _latest(bag["scan"], t_ns)
        obs = np.zeros((0, 2))
        if cloud is not None:
            p = pc2.read_points_numpy(cloud, field_names=("x", "y", "z"),
                                      skip_nans=True)
            if p.size:
                obs = np.asarray(p, dtype=float).reshape(-1, 3)[:, :2]

        def _poly(msg):
            if msg is None or not msg.poses:
                return None
            return np.array([[p.pose.position.x, p.pose.position.y]
                             for p in msg.poses], dtype=float)

        sp = _latest(bag["setpt"], t_ns)
        gl = _latest(bag["goal"], t_ns)
        out.append(Frame(
            t=(t_ns - t0) * 1e-9,
            x0=x0,
            obstacles=obs,
            path=_poly(_latest(bag["path"], t_ns)),
            pred=_poly(_latest(bag["pred"], t_ns)),
            setpoint=None if sp is None else np.array(
                [sp.pose.position.x, sp.pose.position.y]),
            goal=None if gl is None else np.array(
                [gl.pose.position.x, gl.pose.position.y]),
            cost=float(v[1]), solve_ms=float(v[2]),
            success=bool(v[0]), iterations=int(v[13]),
        ))
    return out


def to_scenario(f: Frame, name="bag", margin=2.0):
    """Un Frame come Scenario, cosi' i pannelli non cambiano di una riga."""
    import common
    pts = [f.x0[:2]]
    if f.goal is not None:
        pts.append(f.goal)
    if len(f.obstacles):
        pts += [f.obstacles.min(0), f.obstacles.max(0)]
    P = np.array(pts)
    ext = (P[:, 0].min() - margin, P[:, 0].max() + margin,
           P[:, 1].min() - margin, P[:, 1].max() + margin)
    goal = f.goal if f.goal is not None else (
        f.path[-1] if f.path is not None else f.x0[:2])
    return common.Scenario(name, f.x0[:3].copy(), f.obstacles,
                           np.asarray(goal, dtype=float), f.path, ext)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    bag = read_bag(sys.argv[1])
    print("messaggi per topic:")
    for k, v in TOPICS.items():
        print(f"  {v:28s} {len(bag[k]):6d}")
    fr = frames(bag)
    ok = sum(f.success for f in fr)
    sms = np.array([f.solve_ms for f in fr])
    it = np.array([f.iterations for f in fr], dtype=float)
    print(f"\ncicli di controllo: {len(fr)}  ({fr[-1].t:.1f} s)")
    print(f"  successi: {100*ok/len(fr):.0f}%")
    print(f"  solve_ms: media {sms.mean():.1f}  p95 {np.percentile(sms,95):.1f}  max {sms.max():.1f}")
    if (it >= 0).any():
        print(f"  iterazioni IPOPT: media {it[it>=0].mean():.1f}  max {int(it.max())}")
    print(f"  punti LiDAR per ciclo: media {np.mean([len(f.obstacles) for f in fr]):.0f}")
    npath = sum(f.path is not None for f in fr)
    print(f"  cicli con riferimento A*: {npath}/{len(fr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def hardest_frame(frs: list) -> int:
    """
    Indice del ciclo piu' impegnativo fra quelli REALMENTE risolti.

    Il criterio ingenuo argmax(cost) sceglie sistematicamente un ciclo fallito:
    quelli hanno cost=inf pur non avendo mai invocato IPOPT, e sono i meno
    informativi (nessun iterato da mostrare, nessun minimo da spiegare).
    Si filtra quindi su success e costo finito, con ripiego progressivo se la
    bag non contiene nemmeno un ciclo risolto.
    """
    cost = np.array([f.cost for f in frs], dtype=float)
    ok   = np.array([bool(f.success) for f in frs])
    good = ok & np.isfinite(cost)
    if not good.any():
        good = np.isfinite(cost)
    if not good.any():
        return 0
    masked = np.where(good, cost, -np.inf)
    return int(np.argmax(masked))
