"""
geodesic_field — distanza dal goal che RISPETTA gli ostacoli gia' visti.

PERCHE'. AStarPlanner sceglie il bersaglio locale con una distanza EUCLIDEA dal
goal globale. Misurato su viz/escape_test.py, con il robot 4 m dentro un vicolo
cieco lungo 12 m, le tre candidate sul bordo della finestra valgono:

    candidata                euclidea   geodetica
    (9.35,  0.00) nel vicolo   3.65 m    28.79 m
    (9.60, -1.60) fuori sud    3.76 m     4.06 m
    (9.72, +1.60) fuori nord   3.65 m     3.98 m

L'euclidea le dichiara equivalenti (~3.7 m) e il pianificatore ne sceglie una a
caso: una su tre lo rimanda nella trappola, e a ogni ripianificazione cambia
idea. E' quello il ciclo limite, non la mancanza di memoria — il robot HA gia'
visto il fondo del vicolo, semplicemente la metrica con cui valuta i bersagli
butta via quell'informazione.

La geodetica la usa: e' un fronte d'onda (Dijkstra) propagato DAL GOAL sulle
celle libere della mappa accumulata. Costa un solo campo scalare per
ripianificazione, senza estrarre percorsi.

SPAZIO NON ESPLORATO = LIBERO. E' la scelta ottimistica standard
dell'esplorazione a frontiera: cio' che non si e' ancora visto potrebbe essere
passabile, e vale la pena andare a guardare. La conseguenza voluta e' che il
campo si CORREGGE da solo man mano che il LiDAR scopre: finche' il fondo del
vicolo e' ignoto la geodetica ci passa attraverso ed e' giusto entrare; appena
il fondo e' visto, la geodetica salta a 29 m e il bersaglio viene scartato.
"""

from __future__ import annotations

import heapq
import math
from statistics import NormalDist

import numpy as np

_SQRT2 = math.sqrt(2.0)
_NEIGH = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2))


def block_radius(grid_std: float, obstacle_threshold: float) -> float:
    """Raggio di blocco implicato dalla griglia gaussiana di A*.

    La probabilita' e' P = 1 - Phi(d/sigma), quindi la soglia tau corrisponde a
    d_block = sigma * Phi^-1(1 - tau). E' la stessa formula documentata in
    planner_params_g1.yaml; ricalcolarla qui tiene il campo geodetico COERENTE
    con cio' che A* considera bloccato, invece di introdurre una seconda nozione
    di ostacolo che diverge in silenzio.
    """
    tau = min(max(float(obstacle_threshold), 1e-6), 1.0 - 1e-6)
    return float(grid_std) * NormalDist().inv_cdf(1.0 - tau)


class GeodesicField:
    """Campo di distanza dal goal, propagato sulla mappa nota.

    Parameters
    ----------
    known_xy    : (K, 2) punti ostacolo accumulati (frame mondo).
    goal_xy     : (2,) goal globale.
    robot_xy    : (2,) posa del robot; serve solo a dimensionare il riquadro.
    reso        : lato cella [m].
    r_block     : raggio di inflazione degli ostacoli [m] (vedi block_radius).
    margin      : margine attorno al riquadro robot+goal+ostacoli noti [m]. Deve
                  essere generoso: il percorso d'uscita da una concavita' esce
                  spesso dal rettangolo che contiene robot e goal, e un riquadro
                  stretto lo dichiarerebbe irraggiungibile.
    """

    def __init__(self, known_xy, goal_xy, robot_xy, reso=0.20,
                 r_block=0.40, margin=6.0, max_cells=400_000):
        known = np.asarray(known_xy, dtype=float).reshape(-1, 2)
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        rx, ry = float(robot_xy[0]), float(robot_xy[1])

        xs = [gx, rx]
        ys = [gy, ry]
        if len(known):
            xs += [known[:, 0].min(), known[:, 0].max()]
            ys += [known[:, 1].min(), known[:, 1].max()]
        self.minx, self.maxx = min(xs) - margin, max(xs) + margin
        self.miny, self.maxy = min(ys) - margin, max(ys) + margin
        self.reso = float(reso)

        nx = int(math.ceil((self.maxx - self.minx) / self.reso)) + 1
        ny = int(math.ceil((self.maxy - self.miny) / self.reso)) + 1
        if nx * ny > max_cells:                    # degrada la risoluzione
            k = math.sqrt(nx * ny / float(max_cells))
            self.reso *= k
            nx = int(math.ceil((self.maxx - self.minx) / self.reso)) + 1
            ny = int(math.ceil((self.maxy - self.miny) / self.reso)) + 1
        self.nx, self.ny = nx, ny

        occ = np.zeros((nx, ny), dtype=bool)
        if len(known):
            r = int(math.ceil(r_block / self.reso))
            ix = np.clip(((known[:, 0] - self.minx) / self.reso).astype(int), 0, nx - 1)
            iy = np.clip(((known[:, 1] - self.miny) / self.reso).astype(int), 0, ny - 1)
            # disco di inflazione, precalcolato una volta
            off = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                   if math.hypot(di, dj) * self.reso <= r_block]
            for di, dj in off:
                a = np.clip(ix + di, 0, nx - 1)
                b = np.clip(iy + dj, 0, ny - 1)
                occ[a, b] = True
        self.occ = occ

        self.D = self._wavefront(gx, gy)

    # ------------------------------------------------------------------

    def _idx(self, x, y):
        i = int((float(x) - self.minx) / self.reso)
        j = int((float(y) - self.miny) / self.reso)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None, None

    def _wavefront(self, gx, gy):
        D = np.full((self.nx, self.ny), np.inf, dtype=float)
        gi, gj = self._idx(gx, gy)
        if gi is None:
            return D
        if self.occ[gi, gj]:
            # Il goal e' dentro l'inflazione di un ostacolo (capita quando sta
            # rasente a un muro): si parte dalla cella libera piu' vicina,
            # altrimenti il campo resterebbe tutto infinito e il meccanismo
            # fallirebbe in silenzio proprio nei casi stretti.
            free = np.argwhere(~self.occ)
            if not len(free):
                return D
            k = np.argmin((free[:, 0] - gi) ** 2 + (free[:, 1] - gj) ** 2)
            gi, gj = int(free[k, 0]), int(free[k, 1])

        D[gi, gj] = 0.0
        pq = [(0.0, gi, gj)]
        reso = self.reso
        occ = self.occ
        nx, ny = self.nx, self.ny
        while pq:
            d, i, j = heapq.heappop(pq)
            if d > D[i, j]:
                continue
            for di, dj, w in _NEIGH:
                a, b = i + di, j + dj
                if a < 0 or a >= nx or b < 0 or b >= ny or occ[a, b]:
                    continue
                nd = d + reso * w
                if nd < D[a, b]:
                    D[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
        return D

    # ------------------------------------------------------------------

    def distance(self, x, y) -> float:
        """Distanza geodetica dal goal, o +inf se irraggiungibile/fuori riquadro."""
        i, j = self._idx(x, y)
        if i is None:
            return math.inf
        return float(self.D[i, j])

    def reachable_fraction(self) -> float:
        """Diagnostica: frazione di celle libere raggiunte dal fronte d'onda."""
        libere = int((~self.occ).sum())
        return float(np.isfinite(self.D).sum()) / libere if libere else 0.0
