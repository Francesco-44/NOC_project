#!/usr/bin/env python3
"""
perception — modello di percezione LIMITATA per l'harness offline.

PERCHE' SERVE. viz/common.closed_loop passa ad A* gli ostacoli NOTI PER INTERO
fin dal primo ciclo. Su geometrie convesse la differenza col robot vero e'
piccola, ma su un ostacolo concavo e' tutto: offline A* sa gia' che il vicolo
e' chiuso e non ci entra, quindi il fallimento che si osserva in MuJoCo — il
robot che entra, scopre il fondo e comincia a rimpallare — non e' riproducibile
e non c'e' niente da misurare.

Qui si modella cio' che il G1 vede DAVVERO:

  portata     max_lidar_range (8 m nel profilo G1);
  occlusione  solo il primo bersaglio lungo ciascun azimut, come un ray-cast;
  memoria     PersistentOccupancyMap del repo, la stessa classe di a_star_node,
              cosi' l'accumulo (e il suo decay) e' quello di produzione.

Cio' che NON si modella, e va tenuto presente leggendo i risultati: rumore di
distanza, la fascia di elevazione (qui il mondo e' 2D, quindi ogni ostacolo e'
alto quanto basta), il ritardo del filtro e il voxel a 0.08 m.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "src", "a_star_mpc_planner")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from a_star_mpc_planner.persistent_map import PersistentOccupancyMap  # noqa: E402


class LimitedLidar:
    """Ray-cast 2D con occlusione, su una nuvola di punti-superficie.

    L'occlusione si ottiene raggruppando i punti per azimut e tenendo, per ogni
    settore, SOLO IL PIU' VICINO. E' l'equivalente discreto del primo colpo del
    raggio: cio' che sta dietro a un muro non viene visto, che e' esattamente la
    proprieta' che rende un vicolo cieco indistinguibile da un corridoio aperto
    finche' non lo si percorre.
    """

    def __init__(self, max_range: float = 8.0, n_bearings: int = 360,
                 min_range: float = 0.30):
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.n_bearings = int(n_bearings)

    def scan(self, pose_xy, obstacles: np.ndarray) -> np.ndarray:
        """(M, 2) punti visibili dal punto dato, in frame mondo."""
        if obstacles is None or len(obstacles) == 0:
            return np.zeros((0, 2))
        d = obstacles - np.asarray(pose_xy, dtype=float)[None, :2]
        r = np.hypot(d[:, 0], d[:, 1])
        m = (r >= self.min_range) & (r <= self.max_range)
        if not m.any():
            return np.zeros((0, 2))
        d, r = d[m], r[m]
        pts = obstacles[m]

        b = np.arctan2(d[:, 1], d[:, 0])
        idx = np.floor((b + np.pi) / (2 * np.pi) * self.n_bearings).astype(int)
        idx = np.clip(idx, 0, self.n_bearings - 1)

        # per ogni settore il piu' vicino: ordinando per raggio decrescente e
        # scrivendo in un array indicizzato per settore, l'ultimo scritto (il
        # piu' vicino) sopravvive.
        order = np.argsort(-r)
        first = np.full(self.n_bearings, -1, dtype=int)
        first[idx[order]] = order
        keep = first[first >= 0]
        return pts[keep]


class PerceivedWorld:
    """LiDAR limitato + memoria persistente: la vista del mondo che ha il robot.

    `known()` restituisce i punti accumulati, ed e' cio' che va passato al
    pianificatore al posto degli ostacoli veri.
    """

    def __init__(self, obstacles: np.ndarray, grid_reso: float = 0.20,
                 max_range: float = 8.0, decay_sec: float = 0.0):
        # decay_sec = 0 -> non si dimentica nulla. E' il caso statico di questi
        # mondi; con decay > 0 il robot dimentica il fondo del vicolo e ci
        # rientra per un motivo DIVERSO dal ciclo limite, confondendo la misura.
        self.truth = np.asarray(obstacles, dtype=float)
        self.lidar = LimitedLidar(max_range=max_range)
        self.memory = PersistentOccupancyMap(grid_reso=grid_reso,
                                            decay_sec=decay_sec)
        self._n_seen = 0

    def observe(self, pose_xy, now: float) -> int:
        vis = self.lidar.scan(pose_xy, self.truth)
        if len(vis):
            pts3 = np.hstack([vis, np.zeros((len(vis), 1))])
            self.memory.update(pts3, now)
        self._n_seen = len(vis)
        return self._n_seen

    def known(self) -> np.ndarray:
        """(K, 2) tutto cio' che il robot ha visto finora."""
        big = 1e6
        pts = self.memory.get_points_in_window(-big, -big, big, big)
        return np.zeros((0, 2)) if pts is None else np.asarray(pts)[:, :2]

    @property
    def coverage(self) -> float:
        """Frazione della geometria vera gia' scoperta — utile per capire se un
        fallimento e' d'ignoranza o di decisione."""
        if not len(self.truth):
            return 1.0
        return min(1.0, self.memory.size * 1.0 / len(self.truth))
